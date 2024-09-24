directory=$1
if [ -z "$directory" ]; then
    echo "Usage: merge_slatelog_recursively.sh <directory>"
    exit 1
fi
output_filename="merged.slatelog"
find "${directory}" -type f -name "*.slatelog" -exec cat {} + > ${output_filename}
echo "* output: ${output_filename}"