import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QFileDialog, 
                             QMenu, QMessageBox)
from PyQt5.QtGui import QPainter, QPen, QImage
from PyQt5.QtCore import Qt, QPoint

class DrawingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        # Initialize canvas
        self.image = QImage(800, 600, QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.drawing = False
        self.last_point = QPoint()
        
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
    
    def new_canvas(self):
        self.image.fill(Qt.white)
        self.update()
    
    def save_image(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", 
                                                  "BMP Files (*.bmp);;All Files (*)")
        if file_path:
            self.image.save(file_path)
    
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", 
                                                  "BMP Files (*.bmp);;All Files (*)")
        if file_path:
            self.image.load(file_path)
            self.update()
    
    def paintEvent(self, event):
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(self.rect(), self.image, self.image.rect())
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
    
    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.black, 3, Qt.SolidLine))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

if __name__ == '__main__':
    app = QApplication(sys.argv)
    drawing_app = DrawingApp()
    sys.exit(app.exec_()) 