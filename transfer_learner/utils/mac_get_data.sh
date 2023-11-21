

brew install wget
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zxFDf6wqacM4EfMzLp2YD7kI9g-3KtJk' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zxFDf6wqacM4EfMzLp2YD7kI9g-3KtJk" -O UR5_images_2.zip && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_xY4bdY_JRHJ_R0g7gtvGDvYE1W5YNS1' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_xY4bdY_JRHJ_R0g7gtvGDvYE1W5YNS1" -O UR5_positions.csv && rm -rf /tmp/cookies.txt
mv ./UR5_positions.csv ./transfer_learner/source/data/
unzip UR5_images_2.zip
rm UR5_images_2.zip
mv ./UR5_images_2/ ./transfer_learner/source/data/
mv ./transfer_learner/source/data/UR5_images_2 ./transfer_learner/source/data/images