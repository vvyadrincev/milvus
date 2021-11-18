{ pkgs ? (import <textapp-pkgs> {}) }:
with pkgs;
mkShell {
  name = "vec_indexer";

  hardeningDisable = [ "all" ];

  nativeBuildInputs = [cmake];
  buildInputs = [
    faiss
    boost
    zeromq
    cppzmq
    nlohmann_json
    jaeger-client-cpp
    cuda
    
    nixpkgs.libcbor
    nixpkgs.sparsehash
    nixpkgs.sqlite
  ];


  shellHook =
  ''
    echo "%compile_commands.json" > .ccls
    stdlibpath=${stdenv.cc.cc.outPath}/include/c++/${stdenv.cc.cc.version}
    echo "-isystem" >> .ccls
    echo "$stdlibpath" >> .ccls
    echo "-isystem" >> .ccls
    echo "$stdlibpath/x86_64-unknown-linux-gnu" >> .ccls
    echo "-isystem" >> .ccls
    echo ${stdenv.cc.libc_dev.outPath}/include >> .ccls
    tr -s ' ' '\n' <<< "$NIX_CFLAGS_COMPILE" >> .ccls
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia_manual/:$LD_LIBRARY_PATH
  '';

}
