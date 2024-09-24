FROM golang:1.21-alpine
RUN apk update && apk add --no-cache iproute2
WORKDIR /app
COPY . .
WORKDIR /app/client
RUN go mod tidy
RUN go build -o client .
CMD ["/app/client/client"]
EXPOSE 8080