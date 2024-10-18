// ==UserScript==
// @name        微信公众号文章阅读页面优化
// @namespace   Violentmonkey Scripts
// @match       https://mp.weixin.qq.com/s*
// @grant       none
// @version     1.0
// @author      -
// @description 2024/10/18 08:51:46
// @license MIT
// ==/UserScript==

// 等待网页完成加载
window.addEventListener('load', function () {
    // 加载完成后执行的代码
    setTimeout(() => {
        document.querySelector('#js_pc_qr_code').setAttribute("style", "display: none !important;")
    }, 1000)
    document.querySelector('.rich_media_area_primary_inner').setAttribute('style', "max-width: 61%;")
}, false);
