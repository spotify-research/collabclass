# StackOverflow dataset

Download `stackoverflow.com-Posts.7z` on
<https://archive.org/details/stackexchange> (~15 GB compressed, ~73 GB
unccompressed). Once unzipped, run

    ./parse-stackexchange.py questions Posts.xml
    ./parse-stackexchange.py answers Posts.xml

This will create the file `workspace.pkl` needed to run the notebooks.
