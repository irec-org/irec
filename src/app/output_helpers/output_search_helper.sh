sort -k10 output_search.txt | awk 'match($0, /lambda_:[0-9]*\.?[0-9]*/) {
    print substr($0, RSTART+8, RLENGTH-8), $0
}
' |  sort -k8,8 -k11n -k1n > output_search_normalized.txt
