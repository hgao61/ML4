#!/bin/bash
date = $(date -d "$((RANDOM%1+2010))-$((RANDOM%12+1))-$((RANDOM%28+1)) $((RANDOM%23+1)):$((RANDOM%59+1)):$((RANDOM%59+1))" '+%d-%m-%Y %H:%M:%S')

echo $date
