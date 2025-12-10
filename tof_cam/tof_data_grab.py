import argparse
import cv2
import numpy as np
from pypylon import pylon

def create_basler_cam(serial_number: str) -> pylon.InstantCamera:
    """
    Create a Basler camera instance by serial number.

    :param serial_number: (str), Basler camera serial number.
    :return: (pylon.InstantCamera) Basler camera instance.
    """
    # Get the transport layer factory
    tl_factory = pylon.TlFactory.GetInstance()
    # Set the device information
    device = pylon.DeviceInfo()
    device.SetSerialNumber(serial_number)
    # Create the camera
    cam = pylon.InstantCamera(tl_factory.CreateDevice(device))
    return cam

def create_tof_cam():
    """
    Create a ToF camera object by serial number.
    """
    tof_cam_sn = "24945819"
    tof_cam = create_basler_cam(tof_cam_sn)
    return tof_cam

def config_tof_cam_para(cam: pylon.InstantCamera) -> None:
    """
    Configure a ToF camera (Basler blaze-101) parameter after opening the camera.
    """
    # Operating mode: ShortRange: 0 - 1498 mm / LongRange: 0 - 9990 mm
    cam.OperatingMode.Value = "LongRange"
    # Exposure time (us)
    cam.ExposureTime.Value = 100.0
    # Max depth / Min depth (mm)
    cam.DepthMax.Value = 1498
    cam.DepthMin.Value = 0
    # Fast mode
    cam.FastMode.Value = True
    # Filter spatial
    cam.FilterSpatial.Value = True
    # Filter temporal
    cam.FilterTemporal.Value = True
    # Filter temporal strength
    if cam.FilterTemporal.Value:
        cam.FilterStrength.Value = 200
    # Outlier removal
    cam.OutlierRemoval.Value = True
    # Confidence Threshold (0 - 65536)
    cam.ConfidenceThreshold.Value = 32  # 32 or 3680
    print(f"ToF cam INFO - Operating mode: {cam.OperatingMode.Value} / Exposure time: {cam.ExposureTime.Value} \
    / Depth max: {cam.DepthMax.Value} / min: {cam.DepthMin.Value} / Confidence threshold: {cam.ConfidenceThreshold.Value}")
    # Gamma correction
    cam.GammaCorrection.Value = True
    # GenDC (Generic Data Container) is used to transmit multiple types of image data,such as depth,
    # intensity, and confidence, in a single, structured data stream, making it
    # ideal for 3D and multi-modal imaging applications.
    cam.GenDCStreamingMode.Value = "Off"

def split_tof_container_data(container) -> dict:
    """
    Split the data component from the grab retrieve data container
    Args:
        container: A grab retrieve as data container

    Returns:
        dict: data_dict{Intensity_Image, Confidence_Map, Point_Cloud}
    """
    data_dict = {
        "Intensity_Image": None,
        "Confidence_Map": None,
        "Point_Cloud": None
    }
    for i in range(container.DataComponentCount):
        data_component = container.GetDataComponent(i)
        if data_component.ComponentType == pylon.ComponentType_Intensity:
            data_dict["Intensity_Image"] = data_component.Array
        elif data_component.ComponentType == pylon.ComponentType_Confidence:
            data_dict["Confidence_Map"] = data_component.Array
        elif data_component.ComponentType == pylon.ComponentType_Range:
            data_dict["Point_Cloud"] = data_component.Array.reshape(data_component.Height, data_component.Width, 3)
        data_component.Release()
    return data_dict

def pcl_to_rawdepth(pcl):
    return pcl[:,:,2]  # Get z data from point cloud

def rawdepth_to_heatmap(rawdepth):
    gray_img = cv2.normalize(rawdepth, None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(255 - gray_img, cv2.COLORMAP_TURBO)
    # heatmap = cv2.applyColorMap(255 - gray_img, cv2.COLORMAP_JET)
    return heatmap

def grab_one_intensity_depth():
    cam = create_tof_cam()
    cam.Open()
    config_tof_cam_para(cam)
    # Configure the data type of ToF camera
    # Open 3d point cloud image
    cam.GetNodeMap().GetNode("ComponentSelector").SetValue("Range")
    cam.GetNodeMap().GetNode("ComponentEnable").SetValue(True)
    cam.GetNodeMap().GetNode("PixelFormat").SetValue("Coord3D_ABC32f")  # Coord3D_C16 / Coord3D_ABC32f
    # Open intensity image
    cam.GetNodeMap().GetNode("ComponentSelector").SetValue("Intensity")
    cam.GetNodeMap().GetNode("ComponentEnable").SetValue(True)
    cam.GetNodeMap().GetNode("PixelFormat").SetValue("Mono16")
    # Close confidence map
    cam.GetNodeMap().GetNode("ComponentSelector").SetValue("Confidence")
    cam.GetNodeMap().GetNode("ComponentEnable").SetValue(False)
    cam.GetNodeMap().GetNode("PixelFormat").SetValue("Confidence16")

    # Grab point cloud data
    grab_result = cam.GrabOne(1000)  # timeout: 1s
    assert grab_result.GrabSucceeded(), "Failed to grab ToF data"
    cam.Close()
    result_container = split_tof_container_data(grab_result.GetDataContainer())
    return result_container["Intensity_Image"], result_container["Point_Cloud"]

def parse_args():
    parser = argparse.ArgumentParser(description='Grab ToF camera data.')
    parser.add_argument('--intensity', type=str)
    parser.add_argument('--depth', type=str)
    parser.add_argument('--heatmap', type=str)
    return parser.parse_args()

def main(intensity_img_path, depth_img_path, depth_heatmap_path) -> None:

    # Grab one ToF data
    intensity_img, pcl = grab_one_intensity_depth()
    depth_img = pcl_to_rawdepth(pcl)
    # Save intensity image
    cv2.imwrite(intensity_img_path, intensity_img)
    print(f'Saved {intensity_img_path}')
    # Save depth image
    # Grab and save one depth image.
    np.save(depth_img_path, depth_img)
    print(f'Saved {depth_img_path}')
    # Save the depth heatmap
    cv2.imwrite(depth_heatmap_path, rawdepth_to_heatmap(depth_img))
    print(f'Saved {depth_heatmap_path}')


if __name__ == '__main__':
    args = parse_args()
    main(args.intensity, args.depth, args.heatmap)