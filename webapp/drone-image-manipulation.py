from PIL import Image
import piexif

def set_gps_location(image_path, lat, lon):
    # Convert decimal degrees to degrees, minutes, and seconds format
    def to_deg(value, loc):
        d = int(abs(value))
        m = int((abs(value) - d) * 60)
        s = round(((abs(value) - d) * 60 - m) * 60 * 10000)
        return ((d, 1), (m, 1), (s, 10000)), loc

    # Set latitude and longitude in the EXIF format
    lat_deg = to_deg(lat, 'N' if lat >= 0 else 'S')
    lon_deg = to_deg(lon, 'E' if lon >= 0 else 'W')

    # Load image and get EXIF data
    img = Image.open(image_path)
    exif_dict = piexif.load(img.info['exif'])

    # Set GPS data
    exif_dict['GPS'][piexif.GPSIFD.GPSLatitude] = lat_deg[0]
    exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef] = lat_deg[1]
    exif_dict['GPS'][piexif.GPSIFD.GPSLongitude] = lon_deg[0]
    exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef] = lon_deg[1]

    # Save image with modified EXIF data
    exif_bytes = piexif.dump(exif_dict)
    img.save("modified_" + image_path, "jpeg", exif=exif_bytes)

# Example usage
set_gps_location("DJI_1004.JPG", 1.38414888888889, 103.64940000000001)  
