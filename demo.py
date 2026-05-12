import generate_heatmap
import data_utils
import generate_scanpaths
import visualization
import dynamic_visualization
import ipad_data_utils
import cv2

def main():
    video_path = './demo_data/ex_stimulus.mp4'
    ex1 = data_utils.load_data("./demo_data/ex_data_1.csv")
    ex2 = data_utils.load_data("./demo_data/ex_data_2.csv")
    ex3 = data_utils.load_data("./demo_data/ex_data_3.csv")
    aggregate = ex1.copy()
    aggregate.extend(ex2)
    aggregate.extend(ex3)
    
    vis = dynamic_visualization.DynamicVisualization(video_path)
    vis.add_trace("ex1", ex1, color=(255, 0, 255))
    vis.add_trace("ex2", ex2, color=(255, 255, 0))
    vis.add_trace("ex3", ex3, color=(255, 255, 255))
    
    vis.add_heatmap("all", aggregate)
    vis.save("./output.mp4")

if __name__ == "__main__":
  main()