import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QFileDialog, 
                             QMenu, QMessageBox, QScrollArea, QWidget)
from PyQt5.QtGui import QPainter, QPen, QImage, QTransform
from PyQt5.QtCore import Qt, QPoint, QSize

class Canvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QImage(800, 600, QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.drawing = False
        self.last_point = QPoint()
        self.scale_factor = 1.0
        self.setMinimumSize(800, 600)
    
    def paintEvent(self, event):
        canvas_painter = QPainter(self)
        canvas_painter.scale(self.scale_factor, self.scale_factor)
        canvas_painter.drawImage(0, 0, self.image)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = QPoint(event.pos() / self.scale_factor)
    
    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.drawing:
            current_point = QPoint(event.pos() / self.scale_factor)
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.black, 3, Qt.SolidLine))
            painter.drawLine(self.last_point, current_point)
            self.last_point = current_point
            self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
    
    def setImage(self, image):
        self.image = image
        self.setMinimumSize(self.scale_factor * self.image.size())
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
        
        # Create menu
        self.create_menu()
        
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
    
    def new_canvas(self):
        new_image = QImage(800, 600, QImage.Format_RGB32)
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    drawing_app = DrawingApp()
    sys.exit(app.exec_()) 