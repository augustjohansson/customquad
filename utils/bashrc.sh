export PATH=/usr/lib/ccache:$PATH
export CC=/usr/lib/ccache/gcc
export CXX=/usr/lib/ccache/g++
export MAKEFLAGS=-j8
alias ..='cd ..'
alias ....='cd ../..'
alias c=clear
alias l='ls'
alias ll='ls -l'
alias la='ls -la'

# Press ^D IGNOREEOF+1 times to exit
set -o ignoreeof
export IGNOREEOF=1

# Change permissions on the pip cache to avoid warnings
mkdir -p /root/.cache/pip
chown root:root /root/.cache/pip/

# Alias for customquad installation
alias install-customquad='pushd . && \
                          cd /root && \
                          pip3 install . -U && \
                          popd && \
			  export CC="/usr/lib/ccache/g++ -fpermissive"'

alias clear-cache='mkdir -p /root/.cache/fenics && \
                   rm -f /root/.cache/fenics/* && \
		   rm -f /tmp/call_basix* && \
                   rm -f ./*_petsc_* && \
		   rm -rf ./__pycache__'

# Aliases for dev installation

alias install-ufl='pushd . && \
                   cd /root/ufl-custom && \
                   pip3 install -v . -U --force-reinstall && \
                   popd'

alias install-ffcx='pushd . && \
                    cd /root/ffcx-custom && \
                    git checkout august/2023cq && \
                    pip3 install -v . -U --no-deps && \
                    popd'

alias install-all='clear-cache && \
                   install-ufl && \
                   install-ffcx && \
		   install-customquad'
