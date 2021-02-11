{ pkgs ? (import <nixpkgs> {})
, version ? "dev"
}:
let der = import ./default.nix { inherit pkgs version; };
in
pkgs.mkShell {
  name = "milvus-${version}";

  src = null;

  inputsFrom = [ der ];


  lorriHook =
  ''
    echo "%compile_commands.json" > .ccls
    stdlibpath=${pkgs.stdenv.cc.cc.outPath}/include/c++/${pkgs.stdenv.cc.cc.version}
    echo "-isystem" >> .ccls
    echo "$stdlibpath" >> .ccls
    echo "-isystem" >> .ccls
    echo "$stdlibpath/x86_64-unknown-linux-gnu" >> .ccls
    echo "-isystem" >> .ccls
    echo ${pkgs.stdenv.cc.libc_dev.outPath}/include >> .ccls

    tr -s ' ' '\n' <<< "$NIX_CFLAGS_COMPILE" >> .ccls

    #https://github.com/NixOS/nixpkgs/issues/11390
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia/current/:$LD_LIBRARY_PATH
  '';

}
