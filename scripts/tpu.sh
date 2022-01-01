#!/bin/sh -eEu

usage_exit() {
    cat <<EOF
Usage:
    $(basename "${0}") [command] [arguments]

Commands:
    create
    stop <--vm|--tpu>
    start <--vm|--tpu>
    ssh
    delete
EOF
    exit 1
}

if [ -n "${TPU_CONFIG_JSON+x}" ]; then
    if [ -e "$TPU_CONFIG_JSON" ]; then
        PROJECT=$(cat "$TPU_CONFIG_JSON" | jq -r '.project')
        ZONE=$(cat "$TPU_CONFIG_JSON" | jq -r '.zone')
        NAME=$(cat "$TPU_CONFIG_JSON" | jq -r '.name')
    else
        echo "\$TPU_CONFIG_JSON doesn't exist; $TPU_CONFIG_JSON"
        exit 1
    fi
else
    echo "\$TPU_CONFIG_JSON is undefined. You must set it like:"
    echo ""
    echo "    export TPU_CONFIG_JSON=tpu_configs/foo.json"
    exit 1
fi

cmd_create() {
    set -x
    # Tensorflow version may be 1.13
    # c.f., https://github.com/apls777/kaggle-imaterialist2020-model/blob/7466434d719b346b04ea0cde8c121d45c1338ce1/tf_tpu_models/official/mask_rcnn/mask_rcnn_k8s.yaml#L25:w
    ctpu up --project="$PROJECT" \
        --zone="$ZONE" \
        --disk-size-gb=300 \
        --machine-type=n1-standard-8 \
        --name="$NAME" \
        --tf-version=2.7.0 \
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
    [ $tpu -eq 1 ] && set -x && gcloud compute tpus stop "$NAME" --zone="$ZONE"
    [ $vm -eq 1 ] && set -x && gcloud compute instances stop "$NAME" --zone="$ZONE"
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
    [ $vm -eq 1 ] && set -x && gcloud compute instances start "$NAME" --zone="$ZONE" && set +x
    [ $tpu -eq 1 ] && set -x && gcloud compute tpus start "$NAME" --zone="$ZONE" && set +x
}

cmd_ssh() {
    set -x
    gcloud compute ssh "$NAME" --zone="$ZONE"
}

cmd_delete() {
    set -x
    gcloud compute tpus execution-groups delete "$NAME" --zone="$ZONE"
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

gcloud config set project "$PROJECT"

"cmd_$command" "$@"
