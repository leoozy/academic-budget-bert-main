ps -ef | grep 24hb | grep -v grep | cut -c 9-16 | xargs kill -9
