import heapq
import time
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import sys



## Load Character image

def load_image(file_path, size):
    img = Image.open(file_path)
    img = img.resize(size, Image.NEAREST)  # NEAREST keeps the pixel art look
    return ImageTk.PhotoImage(img)

## resize_images
def resize_image(input_path, output_path, size=(25, 25)):
    with Image.open(input_path) as img:
        img = img.resize(size, Image.LANCZOS)  # Updated to use Image.LANCZOS
        img.save(output_path)

resize_image("start_game.png", "start_game_resized.png", size=(25, 25))
resize_image("up.png", "up_resized.png", size=(25, 25))
resize_image("left.png", "left_resized.png", size=(25, 25))
resize_image("down.png", "down_resized.png", size=(25, 25))
resize_image("right.png", "right_resized.png", size=(25, 25))
resize_image("solve.png", "solve_resized.png", size=(25, 25))



##### Astar Test

import heapq

import heapq

class AStarFrontier2:
    def __init__(self, goal):
        self.goal = goal
        self.frontier = []
        self.entry_finder = {}  # Mapping of states to entries in the priority queue
        self.REMOVED = '<removed-task>'  # Placeholder for a removed task
        self.counter = 0  # Unique sequence count

    def add(self, node):
        """Add a new node or update its priority in the frontier."""
        if node.state in self.entry_finder:
            self.remove(node.state)
        count = self.counter
        heuristic_cost = self.priority(node)  # f(n) = g(n) + h(n)
        entry = [heuristic_cost, count, node]
        self.entry_finder[node.state] = entry
        heapq.heappush(self.frontier, entry)
        self.counter += 1

    def remove(self, state):
        """Mark an existing node as removed by its state."""
        entry = self.entry_finder.pop(state)
        entry[-1] = self.REMOVED

    def priority(self, node):
        """Calculate the A* priority: f(n) = g(n) + h(n)."""
        return self.path_cost(node) + self.heuristic(node.state)

    def path_cost(self, node):
        """Calculate the path cost g(n) by tracing back to the start."""
        cost = 0
        while node.parent is not None:
            cost += 1
            node = node.parent
        return cost

    def heuristic(self, state):
        """Manhattan distance as the heuristic h(n)."""
        return abs(state[0] - self.goal[0]) + abs(state[1] - self.goal[1])

    def empty(self):
        """Check if the frontier is empty, cleaning up removed nodes."""
        while self.frontier and self.frontier[0][-1] == self.REMOVED:
            heapq.heappop(self.frontier)
        return not self.frontier

    def remove_node(self):
        """Pop and return the lowest-priority node from the frontier."""
        while self.frontier:
            heuristic_cost, count, node = heapq.heappop(self.frontier)
            if node is not self.REMOVED:
                del self.entry_finder[node.state]
                return node
        raise Exception("Empty frontier")

    def contains_state(self, state):
        """Check if the frontier contains a state."""
        return state in self.entry_finder

####


class AStarFrontier:
    def __init__(self):
        self.frontier = []
        self.entry_finder = {}  # Mapping of state to entry
        self.counter = 0  # Unique sequence count

    def add(self, node, priority=0):
        if node.state in self.entry_finder:
            self.remove(node.state)
        count = self.counter
        entry = [priority, count, node]
        self.entry_finder[node.state] = entry
        heapq.heappush(self.frontier, entry)
        self.counter += 1

    def remove(self, state):
        entry = self.entry_finder.pop(state)
        entry[-1] = None  # Mark as removed

    def empty(self):
        while self.frontier and self.frontier[0][-1] is None:
            heapq.heappop(self.frontier)
        return not self.frontier

    def remove(self):
        while self.frontier:
            priority, count, node = heapq.heappop(self.frontier)
            if node is not None:
                del self.entry_finder[node.state]
                return node
        raise Exception("Empty frontier")

    def contains_state(self, state):
        return state in self.entry_finder



import heapq

class GreedyFrontier:
    def __init__(self, goal):
        self.goal = goal
        self.frontier = []
        self.entry_finder = {}  # Mapping of state to entry
        self.counter = 0  # Unique sequence count

    def add(self, node, priority=0):
        if node.state in self.entry_finder:
            self.remove(node.state)
        count = self.counter
        heuristic_cost = priority  # Use the provided priority as the heuristic cost
        entry = [heuristic_cost, count, node]
        self.entry_finder[node.state] = entry
        heapq.heappush(self.frontier, entry)
        self.counter += 1

    def remove(self, state):
        entry = self.entry_finder.pop(state)
        entry[-1] = None  # Mark as removed

    def empty(self):
        while self.frontier and self.frontier[0][-1] is None:
            heapq.heappop(self.frontier)
        return not self.frontier

    def remove_node(self):
        while self.frontier:
            heuristic_cost, count, node = heapq.heappop(self.frontier)
            if node is not None:
                del self.entry_finder[node.state]
                return node
        raise Exception("Empty frontier")

    def contains_state(self, state):
        return state in self.entry_finder

    def heuristic(self, state):
        # Assuming `state` is a tuple (row, col)
        row, col = state
        goal_row, goal_col = self.goal
        return abs(row - goal_row) + abs(col - goal_col)
    



class Maze():


    def heuristic(self, state):
        # Calculate Manhattan distance as heuristic
        row, col = state
        goal_row, goal_col = self.goal
        return abs(row - goal_row) + abs(col - goal_col)
    def __init__(self, filename):

        with open(filename) as f:
            contents = f.read()
        if contents.count("A") != 1:
            raise Exception("maze must have exactly one start point")
        if contents.count("B") != 1:
            raise Exception("maze must have exactly one goal")

        # Determinar la altura y el ancho del laberinto.
        contents = contents.splitlines()
        self.height = len(contents)
        self.width = max(len(line) for line in contents)

        # Mantener un registro de las paredes
        self.walls = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                try:
                    if contents[i][j] == "A":
                        self.start = (i, j)
                        row.append(False)
                    elif contents[i][j] == "B":
                        self.goal = (i, j)
                        row.append(False)
                    elif contents[i][j] == " ":
                        row.append(False)
                    else:
                        row.append(True)
                except IndexError:
                    row.append(False)
            self.walls.append(row)

        self.solution = None






    # def move_character_to(self, col, row):
    #     """Move the character to the specified (col, row) position."""
    #     self.update_character(col * self.cell_size, row * self.cell_size)



    def neighbors(self, state):
        row, col = state
        candidates = [
            ("up", (row - 1, col)),
            ("down", (row + 1, col)),
            ("left", (row, col - 1)),
            ("right", (row, col + 1))
        ]

        result = []
        for action, (r, c) in candidates:
            if 0 <= r < self.height and 0 <= c < self.width and not self.walls[r][c]:
                result.append((action, (r, c)))
        return result

    def solve(self, frontier):
        start_time = time.perf_counter()  # Start timing

        self.num_explored = 0
        start = Node(state=self.start, parent=None, action=None)
        frontier.add(start)

        self.explored = set()

        while True:
            if frontier.empty():
                raise Exception("no solution")

            node = frontier.remove()
            self.num_explored += 1

            if node.state == self.goal:
                actions = []
                cells = []
                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)
                
                end_time = time.perf_counter()  # End timing
                elapsed_time = end_time - start_time
                print(f"Time taken: {elapsed_time:.6f} seconds")
                return

            self.explored.add(node.state)

            # Add neighbors to the frontier
            for action, state in self.neighbors(node.state):
                if not frontier.contains_state(state) and state not in self.explored:
                    child = Node(state=state, parent=node, action=action)
                    frontier.add(child)



    def solve3(self, frontier):
        start_time = time.perf_counter()  # Start timing

        self.num_explored = 0
        start = Node(state=self.start, parent=None, action=None)
        
        if isinstance(frontier, AStarFrontier):
            frontier.add(start, self.heuristic(self.start))  # Add with heuristic cost for A*
        else:
            frontier.add(start)  # Default for other frontiers

        self.explored = set()

        while True:
            if frontier.empty():
                raise Exception("no solution")

            node = frontier.remove_node()
            self.num_explored += 1

            if node.state == self.goal:
                actions = []
                cells = []
                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)
                
                end_time = time.perf_counter()  # End timing
                elapsed_time = end_time - start_time
                print(f"Time taken: {elapsed_time:.6f} seconds")
                return

            self.explored.add(node.state)

            # Add neighbors to the frontier
            for action, state in self.neighbors(node.state):
                if not frontier.contains_state(state) and state not in self.explored:
                    child = Node(state=state, parent=node, action=action)
                    if isinstance(frontier, GreedyFrontier):
                        priority = self.heuristic(state)
                        frontier.add(child, priority)
                    else:
                        frontier.add(child)
    def solve2(self, frontier):
        start_time = time.perf_counter()  # Start timing

        self.num_explored = 0
        start = Node(state=self.start, parent=None, action=None)
        
        if isinstance(frontier, AStarFrontier):
            frontier.add(start, self.heuristic(self.start))  # Add with heuristic cost for A*
        else:
            frontier.add(start)  # Default for other frontiers

        self.explored = set()

        while True:
            if frontier.empty():
                raise Exception("no solution")

            node = frontier.remove()
            self.num_explored += 1

            if node.state == self.goal:
                actions = []
                cells = []
                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)
                
                end_time = time.perf_counter()  # End timing
                elapsed_time = end_time - start_time
                print(f"Time taken: {elapsed_time:.6f} seconds")
                return

            self.explored.add(node.state)

            # Add neighbors to the frontier
            for action, state in self.neighbors(node.state):
                if not frontier.contains_state(state) and state not in self.explored:
                    child = Node(state=state, parent=node, action=action)
                    if isinstance(frontier, GreedyFrontier):
                        priority = self.heuristic(state)
                        frontier.add(child, priority)
                    else:
                        frontier.add(child)



    #########





class StackFrontier():
    
    def __init__(self):
        self.frontier = []

    def add(self, node):
        self.frontier.append(node)

    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[-1]
            self.frontier = self.frontier[:-1]
            return node


class QueueFrontier(StackFrontier):

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[0]
            self.frontier = self.frontier[1:]
            return node



class Node():
    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent
        self.action = action


class MazeApp:
    def __init__(self, maze_files):
        self.maze_files = maze_files
        self.current_level = 0
        self.cell_size = 25
        self.player_pos = (0, 0)
        
        # Example goal position, modify this as needed
        self.goal = (9, 9)  # Replace with actual goal coordinates for your maze

        # Create the main window
        self.window = tk.Tk()
        self.window.title("Maze Game")
        
        # Set the background to dark
        self.window.configure(bg='#1c1c1c')  # Dark background color

        # Create GUI components
        self.create_gui()

        # Example images, adapt them as needed
        self.character_img = load_image("pixel_character.png", size=(25, 25))  
        self.character_x = self.player_pos[1] * 25  
        self.character_y = self.player_pos[0] * 25  
        self.grass_img = load_image("grass.png", size=(25, 25))  
        self.tree_img = load_image("tree.png", size=(25, 25))  
        self.path_trace_img = load_image("trace.png", size=(25, 25))  
        self.goal_img = load_image("goal.png", size=(25, 25))  
        self.grass2_img= load_image("./GreenButton-Active.png", size=(25, 25))  

        self.sprite_sheet = Image.open("pixel_character.png")
        self.frame_size = (48, 64)  # Assuming each frame is 25x25 pixels
        self.frames = self.load_frames()
        self.current_frame = 0
        
        # Initialize player position
        self.character_x = self.player_pos[1] * 25
        self.character_y = self.player_pos[0] * 25


    def load_frames(self):
        frames = []
        # Assuming the sprite sheet is laid out in a grid
        sheet_width, sheet_height = self.sprite_sheet.size
        cols = sheet_width // self.frame_size[0]
        rows = sheet_height // self.frame_size[1]

        for row in range(rows):
            for col in range(cols):
                left = col * self.frame_size[0]
                top = row * self.frame_size[1]
                right = left + self.frame_size[0]
                bottom = top + self.frame_size[1]
                frame = self.sprite_sheet.crop((left, top, right, bottom))
                frames.append(ImageTk.PhotoImage(frame))

        return frames



    def update_character(self, x, y):
        """
        Update the position of the character on the canvas.
        """


        self.canvas.delete("character")  # Clear old character position
        self.character_x = x
        self.character_y = y
        # self.canvas.create_image(x, y, anchor=tk.NW, image=self.character_img, tags="character")

        self.canvas.create_image(x, y, anchor=tk.NW, image=self.frames[self.current_frame], tags="character")

    def start_game(self):
        selected_level = self.level_var.get()
        self.current_level = self.maze_files.index(selected_level)
        self.load_level(self.current_level)

    def load_level(self, level_index):
        self.maze = Maze(self.maze_files[level_index])
        self.player_pos = self.maze.start
        self.update_character(self.character_x, self.character_y)  # Initial character position
        self.canvas.config(width=self.maze.width * 25, height=self.maze.height * 25)
        self.draw_maze()

    def clear_maze(self):
        self.canvas.delete("all")

    def next_level(self):
        if self.current_level < len(self.maze_files) - 1:
            self.current_level += 1
            messagebox.showinfo("Level Complete", "Proceeding to the next level!")
            self.load_level(self.current_level)

        else:
            messagebox.showinfo("Game Complete", "You completed all the levels!")
            self.window.quit()

    def create_gui(self):
        # Create Canvas for maze display
        self.canvas = tk.Canvas(self.window, width=800, height=600, bg='#333333')  # Dark canvas background
        self.canvas.pack()

        # Create control frame
        self.control_frame = tk.Frame(self.window, bg='#1c1c1c')  # Dark background for control frame
        self.control_frame.pack()

        # Style configuration for dark theme
        style = ttk.Style()
        style.theme_use("default")  # Use the default theme
        style.configure('TButton', font=('Helvetica', 12), padding=10, background='#333333', foreground='#ffffff')  # Dark button
        style.map('TButton', background=[('active', '#444444')], foreground=[('active', '#ffffff')])  # Hover effect

        # Load images for buttons
        start_img = ImageTk.PhotoImage(Image.open("start_game_resized.png"))
        up_img = ImageTk.PhotoImage(Image.open("up_resized.png"))
        left_img = ImageTk.PhotoImage(Image.open("left_resized.png"))
        down_img = ImageTk.PhotoImage(Image.open("down_resized.png"))
        right_img = ImageTk.PhotoImage(Image.open("right_resized.png"))
        solve_img = ImageTk.PhotoImage(Image.open("solve_resized.png"))

        # Create a menu to select levels
        self.level_var = tk.StringVar()
        self.level_var.set(self.maze_files[0])

        self.level_menu = tk.OptionMenu(self.control_frame, self.level_var, *self.maze_files)
        self.level_menu.config(bg='#333333', fg='#ffffff', activebackground='#444444', activeforeground='#ffffff')
        self.level_menu.grid(row=0, column=0, padx=5, pady=5)

        # Create buttons
        self.start_button = ttk.Button(self.control_frame, text="Start Game", image=start_img, compound="left", command=self.start_game)
        self.start_button.image = start_img  # Keep a reference to avoid garbage collection
        self.start_button.grid(row=0, column=1, padx=5, pady=5)

        # Control buttons with images
        self.up_button = ttk.Button(self.control_frame, image=up_img, command=lambda: self.move("up"))
        self.up_button.image = up_img
        self.up_button.grid(row=1, column=1, padx=5, pady=5)

        self.left_button = ttk.Button(self.control_frame, image=left_img, command=lambda: self.move("left"))
        self.left_button.image = left_img
        self.left_button.grid(row=2, column=0, padx=5, pady=5)

        self.down_button = ttk.Button(self.control_frame, image=down_img, command=lambda: self.move("down"))
        self.down_button.image = down_img
        self.down_button.grid(row=2, column=1, padx=5, pady=5)

        self.right_button = ttk.Button(self.control_frame, image=right_img, command=lambda: self.move("right"))
        self.right_button.image = right_img
        self.right_button.grid(row=2, column=2, padx=5, pady=5)

        self.solve_button = ttk.Button(self.control_frame, text="Solve", image=solve_img, compound="left", command=self.solve_maze)
        self.solve_button.image = solve_img
        self.solve_button.grid(row=1, column=2, padx=5, pady=5)



        self.solve_stack_button = ttk.Button(self.control_frame, text="DFS", image=solve_img, compound="left", 
                                             command=lambda: self.solve_maze_and_animate(StackFrontier))
        self.solve_stack_button.image = solve_img
        self.solve_stack_button.grid(row=0, column=2, padx=5, pady=5)


        self.solve_queue_button = ttk.Button(self.control_frame, text="BFS", image=solve_img, compound="left", 
                                             command=lambda: self.solve_maze_and_animate(QueueFrontier))
        self.solve_queue_button.image = solve_img
        self.solve_queue_button.grid(row=0, column=3, padx=5, pady=5)


        self.solve_stack_button = ttk.Button(self.control_frame, text="A*", image=solve_img, compound="left", 
                                             command=lambda: self.solve_maze_and_animate3(AStarFrontier2))
        self.solve_stack_button.image = solve_img
        self.solve_stack_button.grid(row=0, column=4, padx=5, pady=5)


        self.solve_queue_button = ttk.Button(self.control_frame, text="Greedy", image=solve_img, compound="left", 
                                             command=lambda: self.solve_maze_and_animate3(GreedyFrontier))
        self.solve_queue_button.image = solve_img
        self.solve_queue_button.grid(row=0, column=5, padx=5, pady=5)



        self.window.bind("<w>", lambda event: self.move("up"))
        self.window.bind("<a>", lambda event: self.move("left"))
        self.window.bind("<s>", lambda event: self.move("down"))
        self.window.bind("<d>", lambda event: self.move("right"))

    def draw_maze(self):
        """
        Draw the maze on the canvas. Replaces empty cells with grass.
        """


        self.clear_maze()
        solution = self.maze.solution[1] if self.maze.solution is not None else None

        for i, row in enumerate(self.maze.walls):
            for j, col in enumerate(row):
                x0 = j * self.cell_size
                y0 = i * self.cell_size

                if col:
                    # If it's a wall, fill it with a color (or image if desired)
                    # self.canvas.create_rectangle(x0, y0, x0 + self.cell_size, y0 + self.cell_size, fill="#ffffff")
                    # self.canvas.create_image(x0, y0, anchor=tk.NW, image=self.grass_img)
                    self.canvas.create_rectangle(x0, y0, x0 + self.cell_size, y0 + self.cell_size, fill="black")

                    # self.canvas.create_image(x0, y0, anchor=tk.NW, image=self.grass_img)
                    self.canvas.create_image(x0, y0, anchor=tk.NW, image=self.tree_img)
                else:
                    # If it's an empty cell, place grass
                    self.canvas.create_image(x0, y0, anchor=tk.NW, image=self.grass_img)

                if solution is not None and (i, j) in self.maze.explored:
                    # Yellow or a trace image for the solution path
                    self.canvas.create_image(j * 25, i * 25, anchor=tk.NW, image=self.grass2_img)  # Trace image

                if (i, j) == self.maze.start:
                    # Red for the starting position

                    self.canvas.create_image(j * 25, i * 25, anchor=tk.NW, image=self.path_trace_img)  # Trace image
                    
                elif (i, j) == self.maze.goal:
                    # Green for the goal
                    self.canvas.create_image(j * 25, i * 25, anchor=tk.NW, image=self.goal_img)  # Goal image
                # elif (i, j) == self.player_pos:
                #     # Blue for the player's current position
                #     self.canvas.create_image(j * 25, i * 25, anchor=tk.NW, image=self.character_img)  # Player image
                elif solution is not None and (i, j) in solution:
                    # Yellow or a trace image for the solution path
                    self.canvas.create_image(j * 25, i * 25, anchor=tk.NW, image=self.path_trace_img)  # Trace image



        # Update the character's position on top of the maze
        self.update_character((self.player_pos[1]-.5) * 25, (self.player_pos[0] -2) * 25)




                # Now overlay the solution path, start, goal, and player


    def move(self, direction):
            new_pos = self.get_new_position(self.player_pos, direction)
            if new_pos and not self.maze.walls[new_pos[0]][new_pos[1]]:
                self.player_pos = new_pos
                self.character_x = new_pos[1] * 25
                self.character_y = new_pos[0] * 25
                if (direction == "up"):
                    self.current_frame = 0
                    # self.current_frame = (self.current_frame) % len(self.frames)  # Change frame
                elif (direction == "down"):
                    self.current_frame = 6


                elif (direction == "right"):
                    self.current_frame = 3

                elif (direction == "left"):
                    self.current_frame = 9

                self.draw_maze()
                print(direction)
            if self.player_pos == self.maze.goal:
                messagebox.showinfo("Congrats!", "You reached the goal!")
                self.next_level()

    def get_new_position(self, position, direction):
        row, col = position
        if direction == "up":
            return (row - 1, col) if row > 0 else None
        elif direction == "down":
            return (row + 1, col) if row < self.maze.height - 1 else None
        elif direction == "left":
            return (row, col - 1) if col > 0 else None
        elif direction == "right":
            return (row, col + 1) if col < self.maze.width - 1 else None



    def solve_maze_and_animate3(self, frontier_class):
        """Solve the maze using the given frontier (Stack/Queue/AStar/Greedy) and start the animation."""
        frontier = frontier_class(self.maze.goal)
        self.maze.solve3(frontier)

        # Get the solution path (cells) from the maze
        if self.maze.solution:
            self.solution_path = self.maze.solution[1]
            # Animate character along the solution path
            self.animate_solution()
            messagebox.showinfo("Solved", f"Maze solved with {self.maze.num_explored} states explored!")
            self.maze.solve3(frontier)
            self.draw_maze()




    def solve_maze_and_animate2(self, frontier_class):
        """Solve the maze using the given frontier (Stack/Queue/AStar/Greedy) and start the animation."""
        frontier = frontier_class()
        self.maze.solve2(frontier)

        # Get the solution path (cells) from the maze
        if self.maze.solution:
            self.solution_path = self.maze.solution[1]
            # Animate character along the solution path
            self.animate_solution()
            messagebox.showinfo("Solved", f"Maze solved with {self.maze.num_explored} states explored!")
            self.maze.solve2(frontier)
            self.draw_maze()

    ### test 
    def solve_maze_and_animate(self, frontier_class):
        """Solve the maze using the given frontier (Stack/Queue) and start the animation."""
        frontier = frontier_class()
        self.maze.solve(frontier)


        frontier = frontier_class()  # stackfrontier or queuefrontier

        # get the solution path (cells) from the maze
        if self.maze.solution:
            self.solution_path = self.maze.solution[1]
            # Animate character along the solution path
            self.animate_solution()
            self.maze.solve(frontier)
            messagebox.showinfo("Solved", f"Maze solved with {self.maze.num_explored} states explored!")
            self.draw_maze()
            self.player_pos = self.maze.start

    def animate_solution(self):
        """Animate the character following the solution path."""
        if self.solution_path:
            next_step = self.solution_path.pop(0)
            self.move_character_to(next_step[1], next_step[0])
            # print(self.player_pos, next_step)

            row_diff = next_step[0] - self.player_pos[0]
            col_diff = next_step[1] - self.player_pos[1]
            self.player_pos = next_step

            if row_diff == 0 and col_diff == 1:
                self.move("right")
            elif row_diff == 0 and col_diff == -1:
                self.move("left")
            elif row_diff == -1 and col_diff == 0:
                self.move("up")
            elif row_diff == 1 and col_diff == 0:
                self.move("down")


            # self.move("right")
            # self.move(next_step[1])
            # self.draw_maze()
            # Schedule the next step in the animation
            self.window.after(300, self.animate_solution)  # Adjust speed as needed

    def move_character_to(self, col, row):
        """Move the character to the specified (col, row) position."""
        self.update_character((col - .5) * self.cell_size , (row - 2) * self.cell_size)


    def solve_maze(self):
        self.maze.start = self.player_pos
        frontier_choice = messagebox.askquestion("Solve", "Choose 'yes' for StackFrontier, 'no' for QueueFrontier")
        frontier = StackFrontier() if frontier_choice == "yes" else QueueFrontier()

        try:
            self.maze.solve(frontier)
            messagebox.showinfo("Solved", f"Maze solved with {self.maze.num_explored} states explored!")
            self.draw_maze()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run(self):
        self.window.mainloop()




if len(sys.argv) < 2:
    sys.exit("Usage: python maze_gui.py maze1.txt maze2.txt ...")

maze_files = sys.argv[1:]
app = MazeApp(maze_files)
app.run()
