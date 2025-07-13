# --- Find OpenCV ---
message(STATUS "Looking for OpenCV...")

find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)

if(OpenCV_FOUND)
    message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
    # OpenCV's own find script already creates the OpenCV::opencv target
    # so we don't need to create it manually.
else()
    message(FATAL_ERROR "OpenCV not found. Please install it or set OpenCV_DIR.")
endif() 