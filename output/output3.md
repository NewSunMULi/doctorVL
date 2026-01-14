我们要求函数 $ y = \sin(e^x) $ 的一阶导数。

使用链式法则：

设：
- $ u = e^x $，则 $ y = \sin(u) $
- 则 $ \frac{dy}{dx} = \frac{d}{dx}[\sin(u)] = \cos(u) \cdot \frac{du}{dx} $

计算 $ \frac{du}{dx} $：
- $ \frac{du}{dx} = \frac{d}{dx}(e^x) = e^x $

因此：
$$
\frac{dy}{dx} = \cos(e^x) \cdot e^x
$$

**最终答案：**
$$
\boxed{y' = e^x \cos(e^x)}
$$