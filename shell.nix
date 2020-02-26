{ pkgs ? (import <nixpkgs> {})
, version ? "dev"
}:
let der = import ./default.nix { inherit pkgs version; };
in
pkgs.mkShell {
  name = "milvus-${version}";

  src = null;

  inputsFrom = [ der ];


  shellHook =
  ''
    echo "%compile_commands.json" > .ccls
    tr -s ' ' '\n' <<< "$NIX_CFLAGS_COMPILE" >> .ccls
  '';

}
