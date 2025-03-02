# Quantifying-Mel is for quantifying the melanin content by processing zebrafish images.
Procedure: prepare the images; generate mask using sam; select mask by setting conditions or manually; calculate the melanin ratio by grayscale of mask using equation quoted in paper
1.white balance process_wb.py
2.generating and selecting mask of zebrafish_bestmask.py
3.manully select mask & calculating melanin ratio_allmask.py
4.collecting ratio data to excel_ratio.py
5.put mask(.json) and image in a folder to calculate melanin ratio with self-set threshold and equation_ui.py
