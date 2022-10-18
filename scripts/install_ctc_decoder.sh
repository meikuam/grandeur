#!/usr/bin/env bash

mkdir externals
cd ./externals || { echo "Error"; exit 1; }

# other decoders:
# https://github.com/openspeech-team/openspeech/tree/main/openspeech/decoders
# - lstm attention
# - opensearch decoder
# - rnn transducer decoder
# - transformer decoder
# - transformer transducer decoder


# Install baidu's beamsearch_with_lm
if [ ! -d ctc_decoders ]; then
    git clone https://github.com/usimarit/ctc_decoders.git

    cd ./ctc_decoders || exit
    chmod a+x setup.sh
    chown "$USER":"$USER" setup.sh
    ./setup.sh

    cd ..
fi

cd ..