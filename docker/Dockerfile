FROM dolfinx/dolfinx:v0.6.0-r1

RUN pip3 install --no-cache-dir --force-reinstall --no-deps git+https://github.com/augustjohansson/ffcx-custom.git

RUN pip3 install --no-cache-dir --force-reinstall --no-deps git+https://github.com/augustjohansson/ufl-custom.git

RUN pip3 install --no-cache-dir cppyy pdbpp

RUN git clone --depth=1 https://github.com/algoim/algoim.git && \
    cp -r algoim /usr/local/include/

RUN apt update && \
    apt install -y --no-install-recommends --no-install-suggests ccache parallel && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip3 install --no-cache-dir git+https://github.com/augustjohansson/customquad.git
