#!/usr/bin/env bash
DATADIR=data/omniglot
FILEPATH=$DATADIR/chardata.mat
mkdir -p $DATADIR
wget -c -O $FILEPATH https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat
python mat2pkl.py $FILEPATH
mv ./data.pkl $DATADIR/
rm $FILEPATH