{
  lib,
  stdenv,
  honchoLib,
  python,
  makeWrapper,
  pythonSet,
  src,
}:

stdenv.mkDerivation {
  pname = "honcho";
  version = "3.0.7";

  inherit src;

  dontUnpack = true;
  dontBuild = true;
  dontFixup = true;

  nativeBuildInputs = [ makeWrapper python ];

  installPhase = ''
    mkdir -p $out/lib/python3.13/site-packages $out/bin $out/scripts

    # Copy the uv2nix-built library
    cp -r --no-preserve=mode ${honchoLib}/lib/python3.13/site-packages/* \
      $out/lib/python3.13/site-packages/

    # Copy scripts from source (provision_db.py)
    cp -r $src/scripts/* $out/scripts/

    # Wrap python with the right PYTHONPATH
    makeWrapper ${python}/bin/python $out/bin/python \
      --set PYTHONPATH $out/lib/python3.13/site-packages \
      --set-default HONCHO_CONFIG /etc/honcho/config.toml

    # Symlink fastapi CLI (comes from the fastapi[standard] dep)
    ln -s ${pythonSet.fastapi}/bin/fastapi $out/bin/fastapi

    # Wrap remaining entry points from site-packages bin dir
    if [ -d "${honchoLib}/bin" ]; then
      for f in ${honchoLib}/bin/*; do
        name=$(basename "$f")
        makeWrapper "$f" "$out/bin/$name" \
          --set PYTHONPATH $out/lib/python3.13/site-packages
      done
    fi
  '';

  meta = {
    description = "Infrastructure for AI agents with memory and social cognition";
    homepage = "https://honcho.dev";
    license = lib.licenses.asl20;
    maintainers = with lib.maintainers; [ ];
    platforms = lib.platforms.linux;
  };
}
