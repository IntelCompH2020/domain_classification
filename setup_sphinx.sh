#!/bin/bash

DOCSDIR=docs
PROJECT_NAME=IntelComp_Domain_Classifier
AUTHORS="Jesús Cid, Manuel A. Vázquez"
RELEASE_YEAR=2022

mkdir $DOCSDIR

pushd $DOCSDIR

(echo y ; echo "$PROJECT_NAME" ; echo "$AUTHORS" ; echo $RELEASE_YEAR ; echo en)  | sphinx-quickstart

popd

cp sphinx_settings/* $DOCSDIR/source