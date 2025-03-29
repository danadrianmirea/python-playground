import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
from collections import deque
from datetime import datetime

class TextEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Editor")
        self.root.geometry("800x600")
        
        # Initialize undo stack and history
        self.undo_stack = []
        self.history = deque(maxlen=10)  # Keep last 10 states
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="5")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create text widget
        self.text_widget = tk.Text(self.main_frame, wrap=tk.WORD, undo=True)
        self.text_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create scrollbar
        scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.text_widget.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.text_widget['yscrollcommand'] = scrollbar.set
        
        # Configure grid weights
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)
        
        # Create main menu
        self.create_menu()
        
        # Bind text changes to update history
        self.text_widget.bind('<Key>', self.on_text_change)
        
        # Initialize current file path
        self.current_file = None

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self.new_file)
        file_menu.add_command(label="Open", command=self.open_file)
        file_menu.add_command(label="Save", command=self.save_file)
        file_menu.add_command(label="Save As", command=self.save_file_as)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self.text_widget.edit_undo)
        edit_menu.add_command(label="Redo", command=self.text_widget.edit_redo)
        edit_menu.add_separator()
        edit_menu.add_command(label="Clear", command=self.clear_text)
        
        # History menu
        history_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="History", menu=history_menu)
        history_menu.add_command(label="View History", command=self.show_history)

    def on_text_change(self, event):
        # Save current state to history
        current_text = self.text_widget.get("1.0", tk.END)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.history.append((timestamp, current_text))

    def new_file(self):
        if messagebox.askyesno("New File", "Do you want to create a new file? Current content will be lost."):
            self.text_widget.delete("1.0", tk.END)
            self.current_file = None
            self.root.title("Text Editor - New File")

    def open_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    self.text_widget.delete("1.0", tk.END)
                    self.text_widget.insert("1.0", content)
                    self.current_file = file_path
                    self.root.title(f"Text Editor - {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not open file: {str(e)}")

    def save_file(self):
        if self.current_file:
            try:
                with open(self.current_file, 'w', encoding='utf-8') as file:
                    file.write(self.text_widget.get("1.0", tk.END))
                messagebox.showinfo("Success", "File saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {str(e)}")
        else:
            self.save_file_as()

    def save_file_as(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(self.text_widget.get("1.0", tk.END))
                self.current_file = file_path
                self.root.title(f"Text Editor - {file_path}")
                messagebox.showinfo("Success", "File saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {str(e)}")

    def clear_text(self):
        if messagebox.askyesno("Clear Text", "Are you sure you want to clear the text?"):
            self.text_widget.delete("1.0", tk.END)

    def show_history(self):
        history_window = tk.Toplevel(self.root)
        history_window.title("Edit History")
        history_window.geometry("600x400")
        
        # Create text widget for history
        history_text = tk.Text(history_window, wrap=tk.WORD)
        history_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add history entries
        for timestamp, content in reversed(list(self.history)):
            history_text.insert(tk.END, f"=== {timestamp} ===\n")
            history_text.insert(tk.END, content[:200] + "..." if len(content) > 200 else content)
            history_text.insert(tk.END, "\n\n")
        
        history_text.config(state=tk.DISABLED)  # Make it read-only

if __name__ == "__main__":
    root = tk.Tk()
    app = TextEditor(root)
    root.mainloop() 