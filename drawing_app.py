import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QFileDialog, 
                             QMenu, QMessageBox, QScrollArea, QWidget, QToolBar,
                             QColorDialog, QSpinBox, QLabel)
from PyQt5.QtGui import QPainter, QPen, QImage, QTransform, QColor, QIcon
from PyQt5.QtCore import Qt, QPoint, QSize, QRect

class Canvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QImage(800, 600, QImage.Format_ARGB32)
        self.image.fill(Qt.white)
        self.drawing = False
        self.last_point = QPoint()
        self.scale_factor = 1.0
        self.setMinimumSize(800, 600)
        # Enable keyboard focus
        self.setFocusPolicy(Qt.StrongFocus)
        # Add drawing properties
        self.foreground_color = Qt.black
        self.background_color = Qt.white
        self.brush_size = 3
        # Selection tool properties
        self.selection_mode = False
        self.selection_start = QPoint()
        self.selection_end = QPoint()
        self.has_selection = False
        # Add image history
        self.image_history = [self.image.copy()]
        self.current_history_index = 0
    
    def paintEvent(self, event):
        canvas_painter = QPainter(self)
        canvas_painter.scale(self.scale_factor, self.scale_factor)
        canvas_painter.drawImage(0, 0, self.image)
        
        # Draw selection rectangle if active
        if self.has_selection:
            canvas_painter.setPen(QPen(Qt.blue, 1, Qt.DashLine))
            canvas_painter.drawRect(self.get_selection_rect())
    
    def get_selection_rect(self):
        return QRect(self.selection_start, self.selection_end).normalized()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.selection_mode:
                self.selection_start = QPoint(event.pos() / self.scale_factor)
                self.selection_end = self.selection_start
                self.has_selection = True
                # Set focus when making a selection
                self.setFocus()
            else:
                self.drawing = True
                self.last_point = QPoint(event.pos() / self.scale_factor)
    
    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            if self.selection_mode and self.has_selection:
                self.selection_end = QPoint(event.pos() / self.scale_factor)
                self.update()
            elif self.drawing:
                current_point = QPoint(event.pos() / self.scale_factor)
                painter = QPainter(self.image)
                painter.setPen(QPen(self.foreground_color, self.brush_size, Qt.SolidLine))
                painter.drawLine(self.last_point, current_point)
                self.last_point = current_point
                self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.selection_mode:
                self.selection_end = QPoint(event.pos() / self.scale_factor)
                self.update()
            else:
                self.drawing = False
                self.save_to_history()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete and self.has_selection:
            # Create a new image with the selection area cleared
            new_image = self.image.copy()
            painter = QPainter(new_image)
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.fillRect(self.get_selection_rect(), Qt.transparent)
            painter.end()
            
            # Convert transparent pixels to background color
            for y in range(new_image.height()):
                for x in range(new_image.width()):
                    if new_image.pixelColor(x, y).alpha() == 0:
                        new_image.setPixelColor(x, y, self.background_color)
            
            self.image = new_image
            self.has_selection = False
            self.update()
            self.save_to_history()
            # Accept the event to prevent it from being passed to parent
            event.accept()
        else:
            super().keyPressEvent(event)
    
    def fill_selection(self):
        painter = QPainter(self.image)
        painter.fillRect(self.get_selection_rect(), self.background_color)
    
    def save_to_history(self):
        # Remove any states after current position
        self.image_history = self.image_history[:self.current_history_index + 1]
        # Add new state
        self.image_history.append(self.image.copy())
        self.current_history_index = len(self.image_history) - 1
    
    def undo(self):
        if self.current_history_index > 0:
            self.current_history_index -= 1
            self.image = self.image_history[self.current_history_index].copy()
            self.update()
    
    def redo(self):
        if self.current_history_index < len(self.image_history) - 1:
            self.current_history_index += 1
            self.image = self.image_history[self.current_history_index].copy()
            self.update()
    
    def setImage(self, image):
        self.image = image
        self.setMinimumSize(self.scale_factor * self.image.size())
        self.save_to_history()  # Save the new image to history
        self.update()
    
    def setScaleFactor(self, scale):
        self.scale_factor = scale
        self.setMinimumSize(self.scale_factor * self.image.size())
        self.update()

class DrawingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        # Create canvas widget
        self.canvas = Canvas()
        
        # Create scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.canvas)
        self.scroll_area.setWidgetResizable(True)
        self.setCentralWidget(self.scroll_area)
        
        # Set up window
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('Simple Drawing App')
        
        # Create menu and toolbar
        self.create_menu()
        self.create_toolbar()
        
        self.show()
    
    def create_menu(self):
        # File menu
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('File')
        
        # New action
        new_action = QAction('New', self)
        new_action.setShortcut('Ctrl+N')
        new_action.triggered.connect(self.new_canvas)
        file_menu.addAction(new_action)
        
        # Save action
        save_action = QAction('Save', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)
        
        # Load action
        load_action = QAction('Load', self)
        load_action.setShortcut('Ctrl+O')
        load_action.triggered.connect(self.load_image)
        file_menu.addAction(load_action)
        
        # Exit action
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)    
        
        # Edit menu
        edit_menu = menu_bar.addMenu('Edit')
        
        # Undo action
        undo_action = QAction('Undo', self)
        undo_action.setShortcut('Ctrl+Z')
        undo_action.triggered.connect(self.canvas.undo)
        edit_menu.addAction(undo_action)
        
        # Redo action
        redo_action = QAction('Redo', self)
        redo_action.setShortcut('Ctrl+Y')
        redo_action.triggered.connect(self.canvas.redo)
        edit_menu.addAction(redo_action)
        
        # View menu
        view_menu = menu_bar.addMenu('View')
        
        # Zoom in
        zoom_in_action = QAction('Zoom In', self)
        zoom_in_action.setShortcut('Ctrl++')
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)
        
        # Zoom out
        zoom_out_action = QAction('Zoom Out', self)
        zoom_out_action.setShortcut('Ctrl+-')
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)
        
        # Reset zoom
        reset_zoom_action = QAction('Reset Zoom', self)
        reset_zoom_action.setShortcut('Ctrl+0')
        reset_zoom_action.triggered.connect(self.reset_zoom)
        view_menu.addAction(reset_zoom_action)
    
    def create_toolbar(self):
        # Create toolbar
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # Foreground color picker action
        fg_color_action = QAction('Foreground Color', self)
        fg_color_action.triggered.connect(self.choose_foreground_color)
        toolbar.addAction(fg_color_action)
        
        # Background color picker action
        bg_color_action = QAction('Background Color', self)
        bg_color_action.triggered.connect(self.choose_background_color)
        toolbar.addAction(bg_color_action)
        
        # Selection tool toggle
        self.selection_action = QAction('Selection Tool', self)
        self.selection_action.setCheckable(True)
        self.selection_action.triggered.connect(self.toggle_selection_tool)
        toolbar.addAction(self.selection_action)
        
        # Brush size selector
        brush_label = QLabel('Brush Size:', self)
        toolbar.addWidget(brush_label)
        
        self.brush_size_spinbox = QSpinBox(self)
        self.brush_size_spinbox.setRange(1, 50)
        self.brush_size_spinbox.setValue(3)
        self.brush_size_spinbox.valueChanged.connect(self.change_brush_size)
        toolbar.addWidget(self.brush_size_spinbox)
    
    def new_canvas(self):
        new_image = QImage(800, 600, QImage.Format_ARGB32)
        new_image.fill(Qt.white)
        self.canvas.setImage(new_image)
    
    def save_image(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", 
                                                  "JPG Files (*.jpg);;BMP Files (*.bmp);;All Files (*)")
        if file_path:
            self.canvas.image.save(file_path)
    
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", 
                                                  "JPG Files (*.jpg);;BMP Files (*.bmp);;All Files (*)")
        if file_path:
            loaded_image = QImage()
            if loaded_image.load(file_path):
                self.canvas.setImage(loaded_image)
                self.reset_zoom()
    
    def zoom_in(self):
        self.canvas.setScaleFactor(self.canvas.scale_factor * 1.25)
    
    def zoom_out(self):
        self.canvas.setScaleFactor(self.canvas.scale_factor * 0.8)
    
    def reset_zoom(self):
        # Calculate scale to fit window while maintaining aspect ratio
        window_size = self.scroll_area.viewport().size()
        image_size = self.canvas.image.size()
        
        width_ratio = window_size.width() / image_size.width()
        height_ratio = window_size.height() / image_size.height()
        
        # Use the smaller ratio to ensure the image fits
        scale = min(width_ratio, height_ratio)
        
        # Apply scale factor but don't go below 0.1x or above 5x
        scale = max(0.1, min(scale, 5.0))
        
        self.canvas.setScaleFactor(scale)
    
    def choose_foreground_color(self):
        color = QColorDialog.getColor(self.canvas.foreground_color, self)
        if color.isValid():
            self.canvas.foreground_color = color
    
    def choose_background_color(self):
        color = QColorDialog.getColor(self.canvas.background_color, self)
        if color.isValid():
            self.canvas.background_color = color
    
    def toggle_selection_tool(self, checked):
        self.canvas.selection_mode = checked
        if not checked:
            self.canvas.has_selection = False
            self.canvas.update()
    
    def change_brush_size(self, size):
        self.canvas.brush_size = size

if __name__ == '__main__':
    app = QApplication(sys.argv)
    drawing_app = DrawingApp()
    sys.exit(app.exec_()) 