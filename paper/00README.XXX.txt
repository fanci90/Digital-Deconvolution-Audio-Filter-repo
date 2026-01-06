###########################################################################
# 00README.XXX — arXiv submission control file
# Purpose:
#   This file tells arXiv how to handle your submission’s TeX source, figures,
#   compilation options, and automatic processing.
#
# Notes:
#   - Lines beginning with ‘#’ are ignored (comments).
#   - Only keep ACTIVE options (uncomment those you need).
#   - File must be named exactly: 00README.XXX
###########################################################################

#---------------------------------------------------------------------------
# Compilation method
#---------------------------------------------------------------------------
# Use one (only ONE) of these:
# processwith: pdflatex          # Standard LaTeX -> PDF workflow (recommended)
# processwith: latex             # Classical TeX (produces DVI first)
# processwith: xelatex           # For Unicode fonts or modern documents
# processwith: lualatex          # For LuaTeX features
# processwith: pdftex            # Low-level control (rarely needed)

#---------------------------------------------------------------------------
# TeX Master File
#---------------------------------------------------------------------------
# Set if your main file is not “main.tex”
# For example:
# \\primary: mypaper.tex

#---------------------------------------------------------------------------
# Figure inclusion options
#---------------------------------------------------------------------------
# Include any special post-processing rules or figure handling:
# figures: all                   # Include all figures found in tar/zip
# nofigures                      # Skip figures (for testing only)
# preferredformats: pdf, png, jpg  # Allowed figure types for compilation

#---------------------------------------------------------------------------
# Bibliography
#---------------------------------------------------------------------------
# If using BibTeX, ensure your .bbl file is generated beforehand.
# The arXiv TeX system will *not* run bibtex for you.
# For example:
# include: references.bbl

#---------------------------------------------------------------------------
# Font handling / package compatibility
#---------------------------------------------------------------------------
# To allow modern packages not installed on arXiv yet:
# texlive: 2023                  # Request the latest TeX Live version
# usehypertex                    # Keeps links active in generated PDF
# usenames                        # Include dvipsnames color definitions

#---------------------------------------------------------------------------
# Compilation passes
#---------------------------------------------------------------------------
# nostamp                         # Remove “arXiv identifier” footer on each page
# nomaketitle                     # Prevents auto-running of \maketitle

#---------------------------------------------------------------------------
# Contact / notes
#---------------------------------------------------------------------------
# Optional note for arXiv staff (seen only internally by them):
# message: “Please compile with pdflatex; figures are embedded.”

#---------------------------------------------------------------------------
# End of file
###########################################################################