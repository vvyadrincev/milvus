{ pkgs ? (import <nixpkgs> {})
, source ? ./.
, version ? "dev"
}:

let
  pkgs = (import <nixpkgs>) {
    config = {
      packageOverrides = pkgs: {
        grpc = pkgs.grpc.override { protobuf = pkgs.protobuf3_9; };
        openblas = pkgs.openblas.override { blas64 = false; };
      };
    };
  };
in
with pkgs;

stdenv.mkDerivation rec {
  name = "milvus-${version}";
  inherit version;
  src = lib.cleanSource source;

  nativeBuildInputs = [ gcc cmake wget ];
  buildInputs = [
    # cudatoolkit_10
    boost169
    zlib
    mysql57
    sqlite
    c-ares c-ares.cmake-config
    openssl.dev
    protobuf3_9
    grpc
    prometheus-cpp
    easyloggingpp
    libyamlcpp
    openblas
    gmock
    opentracing-cpp
  ];
  builder=./core/build.sh;
}
