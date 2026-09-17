# A machine that has never seen LiveInterpreter, flatpak remotes, or a CJK font.
# Debian rather than Fedora on purpose: "a different distribution" is the thing
# the flatpak exists to prove, and the host here is Fedora 43.
FROM debian:trixie-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        flatpak ca-certificates fontconfig dbus dbus-x11 \
    && rm -rf /var/lib/apt/lists/*

# Deliberately NOT installed: fonts-noto-cjk, or any font with a Chinese glyph.
# That absence is half of what this container is for.
RUN useradd -m tester
USER tester
WORKDIR /home/tester
