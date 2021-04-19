#!/bin/sh -eEu
NAME=kaggle-imat2020
ZONE=us-central1-a

usage_exit() {
    cat <<EOF
Usage:
    $(basename "${0}") [command] [arguments]

Commands:
    create [project]
    stop
    delete
EOF
    exit 1
}

cmd_create() {
    set -x
    # Tensorflow version may be 1.13
    # c.f., https://github.com/apls777/kaggle-imaterialist2020-model/blob/7466434d719b346b04ea0cde8c121d45c1338ce1/tf_tpu_models/official/mask_rcnn/mask_rcnn_k8s.yaml#L25:w
    ctpu up --project="$1" \
        --zone=$ZONE \
        --disk-size-gb=300 \
        --machine-type=n1-standard-8 \
        --name=$NAME \
        --tf-version=1.15.5 \
        --tpu-size=v3-8
}

cmd_stop() {
    set -x
    gcloud compute tpus stop $NAME --zone=$ZONE
    gcloud compute instances stop $NAME --zone=$ZONE
}

cmd_delete() {
    set -x
    gcloud compute tpus execution-groups delete $NAME --zone=$ZONE
}

case "$1" in
create | stop | delete)
    command="$1"
    ;;
*)
    usage_exit
    ;;
esac
shift

"cmd_$command" "$@"
