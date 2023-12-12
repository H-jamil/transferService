#!/bin/bash

echo "Installing Python 3.10.1 with pyenv..."
pyenv install 3.10.1
pyenv versions

echo "Setting up pyenv-virtualenv..."
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

