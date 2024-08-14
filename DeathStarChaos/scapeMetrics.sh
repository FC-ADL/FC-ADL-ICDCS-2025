#!/bin/bash

# Function to fetch pod metrics, append to JSON file, and sleep for 5 seconds
fetch_and_append_metrics() {
    OUTPUT_FILE="metrics_output.json"
    DURATION=60  # 1 minute
    INTERVAL=5    # 5 seconds

    echo "[]" > "$OUTPUT_FILE"  # Create an empty JSON file

    END_TIME=$((SECONDS + DURATION))

    while [ $SECONDS -lt $END_TIME ]; do
        POD_METRICS=$(kubectl get --raw /apis/metrics.k8s.io/v1beta1/pods | jq .)
        
        # Append metrics to the JSON file
        echo "$POD_METRICS" | jq -s '.' >> "$OUTPUT_FILE"

        sleep $INTERVAL
    done
}

fetch_and_append_metrics
