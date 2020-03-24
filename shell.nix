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
    tr -s ' ' '\n' <<< "$NIX_CFLAGS_COMPILE" >> .ccls

    #https://github.com/NixOS/nixpkgs/issues/11390
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia/current/:$LD_LIBRARY_PATH
  '';

}
