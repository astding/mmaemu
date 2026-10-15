#!/bin/bash
# The Evo 2 40B checkpoint from Hugging Face: the two published parts in parallel, joined, and checked against the
# SHA-256 of the joined file used on the RTX 5070 Ti (dd299612...) in one pass.
set -u
D=/home/ubuntu/models/evo2_40b; mkdir -p $D && cd $D || exit 1
exec >> download.log 2>&1
echo "start $(date -u +%FT%TZ)"
for p in part0 part1; do
  ( for i in 1 2 3 4 5; do
      curl -sSL -C - --retry 5 -o evo2_40b.pt.$p https://huggingface.co/arcinstitute/evo2_40b/resolve/main/evo2_40b.pt.$p && break
      sleep 10
    done; echo "$p curl rc=$? $(date -u +%FT%TZ) $(stat -c %s evo2_40b.pt.$p)" ) &
done
wait
sum=$(cat evo2_40b.pt.part0 evo2_40b.pt.part1 | tee evo2_40b.pt | sha256sum | cut -c1-64)
echo "joined sha256 $sum $(date -u +%FT%TZ)"
if [ "$sum" = dd299612b1c1cdded0dfdcaf4d16f98fc97458261d80f4d662429f0ccb316bc3 ]; then
  rm evo2_40b.pt.part0 evo2_40b.pt.part1; echo DOWNLOAD_DONE
else
  printf '%s  %s\n' 3b74fa4e6158d49265e3e270ba8869390d064358f8bf3d2af0b3e1772728f485 evo2_40b.pt.part0 \
    bdc4a76e0f23f8295e7061c2f0deff24f723bd916dc4cdc4d9216cac9c2d49d5 evo2_40b.pt.part1 | sha256sum -c
  echo DOWNLOAD_FAILED
fi
