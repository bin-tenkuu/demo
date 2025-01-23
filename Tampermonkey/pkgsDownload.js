// ==UserScript==
// @name        pkgs.org 下载链接转换为超链接
// @namespace   Violentmonkey Scripts
// @match       https://*.pkgs.org/*
// @grant       none
// @version     1.0
// @author      -
// @description 2024/3/12 09:16:55
// ==/UserScript==

function toA() {
    // 在此处执行需要在页面加载完成后执行的代码
    let a = document.getElementById('download')?.nextElementSibling?.querySelector('tbody > tr > td')
    if (a) {
        a.innerHTML = '<a href="' + a.innerText + '">' + a.innerText + '</a>'
    }
}


document.addEventListener("DOMContentLoaded", () => {
    // 在此处执行需要在页面加载完成后执行的代码
    setTimeout(() => {
        console.log("toA");
        toA();
    }, 1000)
});
