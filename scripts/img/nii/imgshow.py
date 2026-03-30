import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib

matplotlib.use('QtAgg')

class NiiViewer:
    def __init__(self, file_path):
        # 加载nii.gz文件
        self.img = nib.load(file_path)
        # 转换为三维numpy数组
        self.data = self.img.get_fdata()
        # 获取数据维度
        self.shape = self.data.shape
        # 当前显示的维度和切片
        self.current_dim = 0
        self.current_slice = 0
        # 创建图形
        self.fig, self.ax = plt.subplots()
        # 调整布局以留出空间给滑块
        plt.subplots_adjust(bottom=0.25)
        # 显示初始切片
        self.update_slice()
        # 创建滑块
        self.create_slider()
        # 创建维度选择按钮
        self.create_dim_buttons()
        # 显示图形
        plt.show()
    
    def update_slice(self):
        # 清除当前轴
        self.ax.clear()
        # 根据当前维度获取切片
        if self.current_dim == 0:  # x维度
            if self.current_slice < self.shape[0]:
                slice_data = self.data[self.current_slice, :, :]
        elif self.current_dim == 1:  # y维度
            if self.current_slice < self.shape[1]:
                slice_data = self.data[:, self.current_slice, :]
        else:  # z维度
            if self.current_slice < self.shape[2]:
                slice_data = self.data[:, :, self.current_slice]
        # 显示切片
        self.im = self.ax.imshow(slice_data, cmap='gray')
        # 设置标题
        dim_names = ['X', 'Y', 'Z']
        self.ax.set_title(f'{dim_names[self.current_dim]} Dimension - Slice {self.current_slice}')
        # 更新颜色条
        if not hasattr(self, 'cbar'):
            self.cbar = plt.colorbar(self.im, ax=self.ax)
        else:
            self.cbar.update_normal(self.im)
        # 刷新图形
        self.fig.canvas.draw_idle()
    
    def create_slider(self):
        # 创建滑块的轴
        slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03])
        # 获取当前维度的最大切片数
        max_slice = self.shape[self.current_dim] - 1
        # 创建滑块
        self.slider = Slider(
            ax=slider_ax,
            label='Slice',
            valmin=0,
            valmax=max_slice,
            valinit=0,
            valstep=1
        )
        # 绑定滑块事件
        self.slider.on_changed(self.slider_update)
    
    def slider_update(self, val):
        # 更新当前切片
        self.current_slice = int(val)
        # 更新显示
        self.update_slice()
    
    def create_dim_buttons(self):
        # 创建按钮的轴
        x_btn_ax = plt.axes([0.2, 0.05, 0.2, 0.04])
        y_btn_ax = plt.axes([0.4, 0.05, 0.2, 0.04])
        z_btn_ax = plt.axes([0.6, 0.05, 0.2, 0.04])
        # 创建按钮
        from matplotlib.widgets import Button
        self.x_btn = Button(x_btn_ax, 'X Dimension')
        self.y_btn = Button(y_btn_ax, 'Y Dimension')
        self.z_btn = Button(z_btn_ax, 'Z Dimension')
        # 绑定按钮事件
        self.x_btn.on_clicked(lambda event: self.change_dimension(0))
        self.y_btn.on_clicked(lambda event: self.change_dimension(1))
        self.z_btn.on_clicked(lambda event: self.change_dimension(2))
    
    def change_dimension(self, dim):
        # 更新当前维度
        self.current_dim = dim
        # 重置切片为0
        self.current_slice = 0
        # 更新滑块
        max_slice = self.shape[self.current_dim] - 1
        self.slider.valmax = max_slice
        self.slider.val = 0
        # 更新滑块的轴范围
        self.slider.ax.set_xlim(self.slider.valmin, self.slider.valmax)
        self.slider.label.set_text('Slice')
        # 更新显示
        self.update_slice()

if __name__ == '__main__':
    file_path = "../../../dataset/image/train/50/P2.nii.gz"
    viewer = NiiViewer(file_path)