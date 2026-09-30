$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode %O %S';
ensure_path( 'TEXINPUTS', '.:..:' );
