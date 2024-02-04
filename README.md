# SIV_project

The project has been developed and tested on Ubuntu 22. Inside settings.py there are some fixed value used in order to analyze the videos of the cagliari stadium given by the university. 

This project addresses the growing need for a comprehensive system that leverages image and video processing techniques to analyze soccer matches efficiently. More in detail has been developed a software which is able to perform player detection, team classification and calculate the position of each player detected in order to compute some soccer analysis. An illustrative example of analysis will be presented, showcasing the creation of a heat map depicting the positions of each team.


In order to run this project follow this instruction:

1. Clone the repository:

    `git clone git@github.com:MattiaRigon/SIV_project.git`

2. Navigate inside the folder:
    
    `cd SIV_project`

3. Activate the virtual enviroment ( be sure of have venv installed ):

    `source env/bin/activate`

4. Create a folder an call it video_input (pay attention to give this name). Paste inside the folder the videos that you want to analyze.
You can also create a subfolder.

5. There are two way to run this project:

    - Running `main.py` you can analyze only one video(thinked for the video given by the university which has 2 camera, each one which is positioned on one midfield) and give you the possibility of seen the image with detection and also the soccer field map view.

        In `main.py` in lines 16: you have to place the file name of your input video. The input video must be inside video_input folder. If you have created a subfolder no problem change nome_file like in the example :

        `nome_file = "/cagliari-chievo/2h-right-5min.avi"`

        
        By default they are enable view, so you can see how players are detected and how they are mapped in the two dimension soccer field image. In order to close an image you can press q on the keyboard and the programm keep going forward. In order to disable the visualization of one view you can comment lines in file :

        - `show_image(image)` to disable the view of the input frame with bounding box on the detected players.

        - `show_image(soccer_field_populated)` to disable the visualization of the two dimension soccer field populated with the player that has been detected.


    - Running `generate_heatmaps.py` you can analyze two video together without the visualization of the detection, at the end of the execution of the program the heatmaps are generated.

        As for the main file the only change that you have to perform is change the name of the files inside the line:

            files = ["/cagliari-chievo/2h-left-5min.avi","/cagliari-chievo/2h-right-5min.avi"]
    
        
        If you don't want to wait the end of the video in order to generate the heatmaps you have to place the line :
    
        `soccer_field.generate_heatmaps()`
         which is outside the function main inside the function main, just before the line         
        `end_time = time.time()`
    
        The code is already setup, just uncomment the first one and comment the second one.

        This setup will make decrease the performance of the analysis.
