
import matplotlib.pyplot as plt
import numpy as np
import cv2


def visualize_path(world, way_point_list, path_list=None, show_figure=True):
    H, W = world.shape
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    scale = 1
    grid = np.repeat(world, scale, axis=0)
    grid = np.repeat(grid, scale, axis=1)
    grid = cv2.cvtColor(grid.astype(np.uint8), cv2.COLOR_RGB2BGR)


    # ax.scatter(state[0], state[1], s = 100, c = 'green')

    if path_list is not None:

        ## set color
        color_list = ['C0'] * len(path_list)
        ax_labels_list = ['']
        ax_color_list = ['C0']

        ## plot path
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 2
            
        init = way_point_list[0]
        # grid = cv2.circle(grid, (int((init[0] + 0.5) * scale), int((init[1] + 0.5) * scale)), 30,
        #                     (0, 0, 255), thickness=-1, lineType=8)
        # cv2.putText(grid, "init",
        #                 (int((init[0] + 0.5 - 0.8) * scale), int((init[1] + 0.5 - 0.3) * scale)),
        #                 font, font_size, (0, 0, 0), 10)
        for i, (path, color) in enumerate(zip(path_list, color_list)):
            path_array = np.array(path)
            # ax.plot(path_array[:,0], path_array[:,1], alpha = 0.3, color = color)
            plot_offset = 0
            ax.plot(path_array[:, 0] + plot_offset, path_array[:, 1] + plot_offset, alpha=0.2, color='r', zorder=1)
            ax.scatter(path_array[:, 0] + plot_offset,
                        path_array[:, 1] + plot_offset,
                        s=20,
                        alpha=0.5,
                        color=color,
                        zorder=2)
            
            goal = way_point_list[i+1]
            # grid = cv2.circle(grid, (int((goal[0] + 0.5) * scale), int((goal[1] + 0.5) * scale)), 30,
            #                     (0, 255, 255), thickness=-1, lineType=8)
            # cv2.putText(grid, f"{i}th waypoint",
            #             (int((goal[0] + 0.5 - 0.8) * scale), int((goal[1] + 0.5 - 0.3) * scale)),
            #             font, font_size, (0, 0, 0), 10)
            # (left, right, bottom, top)
        
        ax.imshow(grid, extent=(-0.5, W - 0.5, H - 0.5, -0.5))

    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([])
    ax.set_yticks([], minor=True)
    # ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1))

    if show_figure:
        plt.show()
    return fig
