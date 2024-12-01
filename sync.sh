# 将remote的数据同步到本地
# rsync -avz song@mbp:/Users/song/code/FRAG/frag3/output/ /Volumes/Fast2T/frag3/output
watch -n 600 rsync -avz --sparse song@d4090:/home/song/code/frag4/output/ /Volumes/Fast2T/frag4/output
