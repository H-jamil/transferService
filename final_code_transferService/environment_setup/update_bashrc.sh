#!/bin/bash

# Define the location of .bashrc
BASHRC="$HOME/.bashrc"

# Define the content to add
CONTENT_TO_ADD="
# Always list directory contents upon 'cd'
cd() { builtin cd \"\$@\"; ll; }

alias cd..='cd ../'                         # Go back 1 directory level (for fast typers)
alias ..='cd ../'                           # Go back 1 directory level
alias ...='cd ../../'                       # Go back 2 directory levels
alias .3='cd ../../../'                     # Go back 3 directory levels
alias .4='cd ../../../../'                  # Go back 4 directory levels
alias .5='cd ../../../../../'               # Go back 5 directory levels
alias .6='cd ../../../../../../'            # Go back 6 directory levels
alias ~=\"cd ~\"                            # ~:            Go Home
alias c='clear'                             # c:            Clear terminal display
"

# Append the content to .bashrc
echo "$CONTENT_TO_ADD" >> "$BASHRC"

# Source .bashrc to apply changes
source "$BASHRC"

echo "Changes added to .bashrc and sourced."
echo "Cloning transferService repository..."
git clone https://github.com/H-jamil/transferService.git
