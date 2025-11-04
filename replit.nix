{ pkgs }: {
  deps = [
    pkgs.python311Full
    pkgs.python311Packages.pip
    pkgs.ffmpeg
    pkgs.portaudio
  ];
}
