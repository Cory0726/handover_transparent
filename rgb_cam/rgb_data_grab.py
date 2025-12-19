import argparse

from pypylon import pylon
import cv2

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

def create_rgb_cam_obj():
    """
    Create a RGB camera object by serial number.
    """
    rgb_cam_sn = "24747625"
    rgb_cam = create_basler_cam(rgb_cam_sn)
    return rgb_cam

def config_rgb_cam_para(cam: pylon.InstantCamera) -> None:
    """
    Configurate RGB camera (acA1300-75gc) parameter after opening the camera.

    Args:
        camera (pylon.InstantCamera): A RGB camera instance
    """
    # Width and height
    cam.Width.Value = 1280
    cam.Height.Value = 1024
    # Pixel format
    cam.PixelFormat.Value = "BayerBG8"
    # Exposure time (Abs) [us]
    cam.ExposureTimeAbs.Value = 7500
    # Exposure auto
    cam.ExposureAuto.Value = "Off"
    # Gain (Raw)
    cam.GainSelector.Value = "All"
    cam.GainRaw.Value = 136
    # Gain auto
    cam.GainAuto.Value = "Off"
    # Balance white auto
    cam.BalanceWhiteAuto.Value = "Off"

def grab_one_rgb_img():
    # Initialize the RGB camera
    cam = create_rgb_cam_obj()
    cam.Open()
    config_rgb_cam_para(cam)

    # Grab one rgb image
    grab_result = cam.GrabOne(1000)  # timeout: 1 s
    if grab_result.GrabSucceeded():
        bayer_img = grab_result.Array
        rgb_img = cv2.cvtColor(bayer_img, cv2.COLOR_BAYER_BG2RGB)  # Convert bayer to RGB
    grab_result.Release()
    cam.Close()
    return rgb_img

def main(rgb_img_path):
    # Grab one rgb image
    rgb_img = grab_one_rgb_img()
    # Save
    cv2.imwrite(rgb_img_path, rgb_img)
    print(f'Saved {rgb_img_path}')

def parse_args():
    parser = argparse.ArgumentParser(description='Grab one RGB image.')
    parser.add_argument('--rgb', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args.rgb)