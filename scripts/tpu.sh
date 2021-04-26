#!/bin/sh -eEu
NAME=kaggle-imat2020
ZONE=us-central1-a

usage_exit() {
    cat <<EOF
Usage:
    $(basename "${0}") [command] [arguments]

Commands:
    create [project]
    stop <--vm|--tpu>
    start <--vm|--tpu>
    ssh
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
    vm=1
    tpu=1
    while test $# -ne 0; do
        case "$1" in
        --vm)
            tpu=0
            ;;
        --tpu)
            vm=0
            ;;
        *)
            echo "invalid option; $1"
            usage_exit
            ;;
        esac
        shift
    done
    [ $tpu -eq 1 ] && set -x && gcloud compute tpus stop $NAME --zone=$ZONE
    [ $vm -eq 1 ] && set -x && gcloud compute instances stop $NAME --zone=$ZONE
}

cmd_start() {
    vm=1
    tpu=1
    while test $# -ne 0; do
        case "$1" in
        --vm)
            tpu=0
            ;;
        --tpu)
            vm=0
            ;;
        *)
            echo "invalid option; $1"
            usage_exit
            ;;
        esac
        shift
    done
    [ $vm -eq 1 ] && set -x && gcloud compute instances start $NAME --zone=$ZONE && set +x
    [ $tpu -eq 1 ] && set -x && gcloud compute tpus start $NAME --zone=$ZONE && set +x
}

cmd_ssh() {
    set -x
    gcloud compute ssh $NAME --zone=$ZONE
}

cmd_delete() {
    set -x
    gcloud compute tpus execution-groups delete $NAME --zone=$ZONE
}

case "$1" in
create | stop | start | ssh | delete)
    command="$1"
    ;;
*)
    usage_exit
    ;;
esac
shift

"cmd_$command" "$@"
