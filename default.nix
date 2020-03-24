{ pkgs ? (import <nixpkgs> {})
, version ? "dev"
}:

let
  pkgs = (import <nixpkgs>) {
    config = {
      packageOverrides = pkgs: {
        openblas = pkgs.openblas.override { blas64 = false; };
        aws-sdk-cpp = pkgs.aws-sdk-cpp.override {customMemoryManagement = false; apis = ["s3"];};
      };
    };
  };
  #pin nixpkgs
  pkgs-grpc-25 =
    let pinned1 = pkgs.fetchFromGitHub {
    owner  = "NixOS";
    repo   = "nixpkgs-channels";
    rev    = "975a6b7b1dbcbd9bba2b0a2d3d2d38bb605ababa";
    sha256 = "1q3r1z163g5xpnalp7p5p24m0xz3ccg1mjsss1i61zqqb4x7fk06";};
    in (import pinned1) {
      config = {
        packageOverrides = pkgs: {
          grpc = pkgs.grpc.override { protobuf = pkgs.protobuf3_9; };
        };
      };
    };
  # get sha with
  # nix-prefetch-url --unpack https://github.com/NixOS/nixpkgs-channels/archive/811448e4ac1aacd7525d0e3c425d746fd659a86b.tar.gz
in with pkgs;

stdenv.mkDerivation rec {
  name = "milvus-${version}";
  inherit version;
  src = if lib.inNixShell then null else ./.;

  nativeBuildInputs = [ gcc cmake wget ];
  buildInputs = [
    cudatoolkit_10
    boost169
    zlib
    mysql57
    sqlite
    #deps of grpc
    #they have to be added to use findGrpc cmake functions
    pkgs-grpc-25.c-ares
    pkgs-grpc-25.c-ares.cmake-config
    pkgs-grpc-25.openssl.dev
    pkgs-grpc-25.protobuf3_9
    pkgs-grpc-25.grpc
    prometheus-cpp
    easyloggingpp
    libyamlcpp
    openblas
    gmock
    opentracing-cpp
    aws-sdk-cpp
    zeromq
    cppzmq
    libcbor
    #faiss python bindings
    swig
    python3
    python37Packages.numpy
    python37Packages.setuptools
    python37Packages.jupyter
  ];
  builder=./core/build.sh;
}
