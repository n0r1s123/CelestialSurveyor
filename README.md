**The goal**

CelestialSurveyor is designed to analyze astronomical images and search moving objects (like asteroids and comets).
My personal use case I tried to make as simple as possible: 
- Astronomer did some imaging session with his telescope. 
- Astronomer loads light and calibration frames without any preliminary processing to CelestialSurveyor.
- Astronomer clicks one button and gets areas of interest where is high probability of asteroid or comet detection and the list of known objects in the given Field Of View marked on the images.
- If there are objects not marked as known objects - Astronomer takes some actions to confirm or not his discovery.
  
I believe that a lot of people interested in astro imaging do not search something new on their images. That's why I worked on this tool to make this process simple.

**Usage examples**

- Run the app.
  
- Provide destinations for calibration frames if there are any. It's strongly recommended to have up-to-date dark frames to avoid hot-pixels impact.
  
<img width="1280" alt="calibration" src="https://github.com/user-attachments/assets/896ca98e-7792-4f5f-b298-17613ae30eba">

- Add light frames by clicking "Add files" button. Make sure that all the light frames have the same exposure (this information will be shown in the file list). If not - uncheck the ones with wrong exposure.
  
<img width="1279" alt="add_files" src="https://github.com/user-attachments/assets/a3386992-c9f0-4574-8366-446c7f703bba">

- Check "Debayer" if you use one-shot color camera. For mono cameras this checkbox MUST be unchecked.
  
- Check "Align images" if the images are not pre-aligned (basically in case if raw files without preprocessing are provided - this checkbox needs to be checked)

- "Non-linear" checkbox should be checked if your images are stretched for some reason. Otherwise in case of images without preprocessing - this checkbox must be unchecked.

Most common case of the checkboxes above is shown on the previous image.

- If everything is fine - click "Load images". Loading will be started. Steps to be done during this step (NOTE: original files will not be changed):
  
1) loading images
2) image calibration
3) plate solving of each frame
4) image alingment basing on plate-solving solution
5) cropping (to avoid black areas after alignment)
6) stretching.

<img width="1278" alt="running_process" src="https://github.com/user-attachments/assets/7087c382-7347-4361-9c1e-7234a0bec1d8">

- Review the light frames by walking through the file list. Gray images will be shown on the right side. Uncheck bad frames.

- Select folder whare results will be stored.

- Choose "Annotation magnitude limit". This value is used in fetch known objects query. Too big values will lead to long request time and there will be too many object annotated on the result image. Choose value that is suitable for your rig.

- Click "Process". 

- After the processing is complete open the folder you specified for results. 

There are files called "results.png" and "results_annotated.png". There are areas of interest found by AI model with probability. Maximum image is used to display results (it meand that each pixel has maximum value from all the images), that's why some of the bright objects can be seen on it. 
Several close results with 1.00 probability mean that most probably there is something moving between stars. 
The second file contains annotations for the known object. Annotations are done for the first image of each imaging session. So if you provided data from 2 nights in a row - probably there will be asteroid marked twice, depends on imaging conditions. 

