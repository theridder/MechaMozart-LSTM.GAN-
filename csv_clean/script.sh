for file in *.csv;
do echo ${file}; 
    csvmidi "${file}" "../midi_clean/${file}";
done
