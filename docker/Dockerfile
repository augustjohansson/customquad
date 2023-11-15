FROM dolfinx/dolfinx:v0.6.0-r1

RUN pip3 install --no-cache-dir git+https://github.com/FEniCS/fiat.git

RUN pip3 install --no-cache-dir --force-reinstall --no-deps git+https://github.com/augustjohansson/ffcx-custom.git@august/2023cq

RUN pip3 install --no-cache-dir --force-reinstall --no-deps git+https://github.com/augustjohansson/ufl-custom.git

RUN pip3 install --no-cache-dir cppyy

RUN git clone --depth=1 https://github.com/algoim/algoim.git && \
    cp -r algoim /usr/local/include

RUN pip3 install --no-cache-dir git+https://github.com/augustjohansson/customquad.git

RUN pip3 install --no-cache-dir pdbpp

RUN apt update && \
    apt install -y --no-install-recommends --no-install-suggests ccache && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
