#!/sur/bin/env bash
cp $HOME/.vimrc .
docker build -t devenv -f Dockerfile ./
