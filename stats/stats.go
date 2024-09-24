package stats

import (
	"fmt"
	"os"
	"sync"
	"time"
	"sort"

	"go.uber.org/zap"
	"gonum.org/v1/gonum/stat"
)

type Sample struct {
	timestamp time.Time
	val       float64
}

type StatsMgr struct {
	statsVals map[string]float64
	mtx       sync.Mutex
	log       *zap.SugaredLogger
	// Every time Sample() is called, we append the snapshot of the current values
	// to this collection.
	sampleCollection map[string][]Sample
}

func NewStatsMgrImpl(logger *zap.SugaredLogger) *StatsMgr {
	ret := &StatsMgr{
		statsVals:        make(map[string]float64),
		sampleCollection: make(map[string][]Sample),
		log:              logger,
	}
	return ret
}

func (s *StatsMgr) Set(k string, val float64, tid uint) {
	s.mtx.Lock()
	s.statsVals[keyWithTid(k, tid)] = val
	s.mtx.Unlock()
}

func (s *StatsMgr) Incr(k string, tid uint) {
	s.mtx.Lock()
	s.statsVals[keyWithTid(k, tid)] += 1.0
	s.mtx.Unlock()
}

func (s *StatsMgr) DirectMeasurement(k string, t time.Time, val float64, tid uint) {
	s.mtx.Lock()
	s.sampleCollection[keyWithTid(k, tid)] =
		append(s.sampleCollection[keyWithTid(k, tid)], Sample{timestamp: t, val: val})
	s.mtx.Unlock()
}

func (s *StatsMgr) sample() {
	now := time.Now()
	for statName, val := range s.statsVals {
		s.sampleCollection[statName] =
			append(s.sampleCollection[statName],
				Sample{timestamp: now, val: val})
		s.statsVals[statName] = 0.0
	}
}

func (s *StatsMgr) DumpStatsToFolder(folderName string) error {
	s.mtx.Lock()
	defer s.mtx.Unlock()

	os.RemoveAll(folderName)
	os.MkdirAll(folderName, os.ModePerm)

	for statName, sampleSlice := range s.sampleCollection {
		filename := fmt.Sprintf("%s/%s.csv", folderName, statName)
		s.log.Infow("creating stats file", "filename", filename)
		file, err := os.Create(filename)
		if err != nil {
			return fmt.Errorf("unable to create file %s: %s", statName, err.Error())
		}

		defer file.Close()

		for _, sample := range sampleSlice {
			_, err := file.Write([]byte(fmt.Sprintf("%d,%f\n", sample.timestamp.UnixNano(), sample.val)))
			if err != nil {
				return fmt.Errorf("unable to write to file %+v: %s", file, err.Error())
			}
		}
		s.log.Infow("finished writing to stats file", "filename", filename)
		if statName == "client.req.latency.0" {
			// calculate mean, p50, p90, p99, p999 latencies and save to a file
			values := make([]float64, len(sampleSlice))
			for i, sample := range sampleSlice {
				values[i] = sample.val
			}

			// Calculate mean
			mean := stat.Mean(values, nil)

			// Sort values for percentile calculations
			sort.Float64s(values)

			// Calculate percentiles using gonum's stat.Quantile
			p50 := stat.Quantile(0.50, stat.Empirical, values, nil)
			p75 := stat.Quantile(0.75, stat.Empirical, values, nil)
			p90 := stat.Quantile(0.90, stat.Empirical, values, nil)
			p99 := stat.Quantile(0.99, stat.Empirical, values, nil)
			p999 := stat.Quantile(0.999, stat.Empirical, values, nil)
			// Write to file
			filename = fmt.Sprintf("%s/latency_percentiles.csv", folderName)
			file, _ = os.Create(filename)
			defer file.Close()
			_, err := file.Write([]byte(fmt.Sprintf("mean,%f\np50,%f\np75,%f\np90,%f\np99,%f\np999,%f\n", mean, p50, p75, p90, p99, p999)))
			if err != nil {
				fmt.Printf("unable to write to file %+v: %s", file, err.Error())
			}
		}
	}

	return nil
}

func (s *StatsMgr) PeriodicStatsCollection_2(period time.Duration, done chan struct{}) {
	ticker := time.NewTicker(period)
	defer ticker.Stop()

	for {
		select {
		case <-done:
			return
		case <-ticker.C:
			s.mtx.Lock()
			s.sample()
			s.mtx.Unlock()
		}
	}
}

func (s *StatsMgr) PeriodicStatsCollection(period time.Duration, done chan struct{}) {
	// defer wg.Done()

	ticker := time.NewTicker(period)
	defer ticker.Stop()

	for {
		select {
		case <-done:
			return
		case <-ticker.C:
			s.mtx.Lock()
			s.sample()
			s.mtx.Unlock()
		}
	}
}

func keyWithTid(k string, tid uint) string {
	return fmt.Sprintf(k+".%d", tid)
}
