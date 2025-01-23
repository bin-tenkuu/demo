// ==UserScript==
// @name        魔塔作弊
// @namespace   Violentmonkey Scripts
// @match       https://h5mota.com/games/*
// @grant       none
// @version     1.0
// @author      -
// @description 2024/5/4 10:37:52
// ==/UserScript==

function hookAfter(raw, after) {
    return function (...args) {
        let ret = raw.call(this, ...args);
        after.call(this, ...args);
        return ret;
    }
}

let cheat = window.cheat = {
    start() {
        this.itemGem = 2
        this.autoBattle?.()
        delete this.start
    },
    set itemGem(count) {
        for (let item of Object.values(core.items.items)) {
            if (item.itemEffect) {
                item.itemEffect = item.itemEffect.replace(/(?<=(?:def) *\+=)(\d+\*)?/g, x => count + "*");
                item.itemEffect = item.itemEffect.replace(/(?<=(?:atk|hp|max) *\+=)(\d+\*)?/g, x => count * 1.5 + "*");
                console.log(item);
            }
        }
    },
    get keys() {
        let values = core.status.hero.items.tools;
        for (let key of Object.keys(values)) {
            if (key.endsWith("Key")) {
                console.log(key, values[key]);
            }
        }
    },
    set keys(count) {
        let values = core.status.hero.items.tools;
        for (let key of Object.keys(values)) {
            if (key.endsWith("Key")) {
                let old = values[key];
                let value = old + count;
                console.log(key, old, value);
                values[key] = value;
            }
        }
    },
    get gems() {
        let values = core.values;
        for (let key of Object.keys(values)) {
            if (key.endsWith("Gem") || key.endsWith("Jewel")) {
                console.log(key, values[key]);
            }
        }
    },
    set gems(count) {
        let values = core.values;
        for (let key of Object.keys(values)) {
            if (key.endsWith("Gem") || key.endsWith("Jewel")) {
                let old = values[key];
                let value = old * count;
                if (key.startsWith('red')) {
                    value *= 1.5;
                }
                console.log(key, old, value);
                values[key] = value;
            }
        }
    },
    get potions() {
        let values = core.values;
        for (let key of Object.keys(values)) {
            if (key.endsWith("Potion")) {
                console.log(key, values[key]);
            }
        }
    },
    set potions(count) {
        let values = core.values;
        for (let key of Object.keys(values)) {
            if (key.endsWith("Potion")) {
                let old = values[key];
                let value = old * count;
                console.log(key, old, value);
                values[key] = value;
            }
        }
    },
    get all() {
        this.gems;
        this.potions;
    },
    set all(count) {
        this.gems = count;
        this.potions = count;
    },
    get enemys() {
        return core.material.enemys;
    },
    set enemys(handler) {
        if (typeof handler === 'function') {
            for (let enemy of Object.values(core.enemys.enemys)) {
                if (enemy.hp == 0) {
                    continue;
                }
                handler(enemy)
                console.log(enemy)
            }
            for (let enemy of Object.values(core.material.enemys)) {
                if (enemy.hp == 0) {
                    continue;
                }
                handler(enemy)
            }
        }
    },
    enemySpecial(num) {
        cheat.enemys = (enemy) => {
            if (!Array.isArray(enemy.special)) {
                return;
            }
            let index = enemy.special.indexOf(num)
            if (index != -1) {
                console.log(enemy)
                enemy.special.splice(index, 1);
            }
        }
    },
    get status() {
        return core.status;
    },
    get hero() {
        return core.status.hero;
    },
    getItem() {
        delete this.getItem;
        core.events.getItem = function (a, g, j, i, c, k) {
            if (g > 0) {
                let cls = core.material.items[a].cls;
                if (cls !== 'equips') {
                    events.prototype.getItem.call(this, a, g, j, i, c, k);
                }
            }
            events.prototype.getItem.call(this, a, g, j, i, c, k);
        }
    },
    __auto__: 3,
    blackItem: {},
    autoBattle() {
        /**
         * --------------- 安装说明 ---------------
         *
         * 首先安装高级动画插件
         * 然后将该插件复制到插件编写里面即可
         * 注意高级动画插件要在本插件之前
         *
         * --------------- 使用说明 ---------------
         *
         * 变量__auto__控制功能，为一个数字。它需要由一下几个变量构成：
         * 1. core.plugin.AUTO_BATTLE
         * 2. core.plugin.AUTO_ITEM
         * 具体使用方法如下：
         * flags.__auto__ = core.plugin.AUTO_BATTLE; // 开启自动清怪，关闭自动拾取
         * flags.__auto__ = core.plugin.AUTO_BATTLE | core.plugin.AUTO_ITEM; // 二者都开启
         * flags.__auto__ = core.plugin.AUTO_ITEM; // 开启自动拾取，关闭自动清怪
         * flags.__auto__ = 0; 关闭所有功能
         * 更多内容可以在插件注释中查看
         */

        const AUTO_BATTLE = 1;
        const AUTO_ITEM = 2;

        const transitionList = [];

        control.prototype.moveOneStep = hookAfter(control.prototype.moveOneStep, update);

        control.prototype.moveDirectly = hookAfter(control.prototype.moveDirectly, update);

        function update() {
            core.auto();
            cheat.afterAuto();
            if (main.replayChecking) return;
            for (let i = 0; i < transitionList.length; i++) {
                const t = transitionList[i];
                let {x, y} = core.status.hero.loc;
                t.value.x = x * 32 - core.bigmap.offsetX;
                t.value.y = y * 32 - core.bigmap.offsetY;
            }
        }

        /**
         * 是否清这个怪，可以修改这里来实现对不同怪的不同操作
         * @param {string} enemy
         * @param {number} x
         * @param {number} y
         */
        function canBattle(enemy, x, y) {
            const loc = `${x},${y}`;
            const floor = core.floors[core.status.floorId];
            const e = core.material.enemys[enemy];
            const hasEvent =
                // has(floor.afterBattle?.[loc]) ||
                // has(floor.beforeBattle?.[loc]) ||
                has(e.beforeBattle) ||
                has(e.afterBattle) ||
                has(floor.events[loc]);
            // 有事件，不清
            if (hasEvent) return false;
            const damage = core.getDamageInfo(enemy, void 0, x, y)?.damage;
            // 0伤或负伤，清
            if (has(damage) && damage <= 0) return true;
            return false;
        }

        /**
         * 判断一个点是否能遍历
         */
        function judge(block, nx, ny, tx, ty, dir, floorId) {
            if (!has(block)) return {};
            const cls = block.event.cls;
            const loc = `${tx},${ty}`;
            const floor = core.floors[floorId];
            const changeFloor = floor.changeFloor[loc];
            const isEnemy =
                cheat.__auto__ & AUTO_BATTLE &&
                cls.startsWith('enemy');
            const isItem =
                cheat.__auto__ & AUTO_ITEM && cls === 'items';

            if (has(changeFloor)) {
                if (!core.noPass(tx, ty, floorId) && !core.canMoveHero(nx, ny, dir)) {
                    return false;
                }
                if (changeFloor.ignoreChangeFloor ?? core.flags.ignoreChangeFloor) {
                    return true;
                }
                return false
            }

            if (has(core.floors[floorId].events[loc])) return false;

            if (isEnemy || isItem)
                return {
                    isEnemy,
                    isItem
                };

            return false;
        }

        /**
         * 是否捡拾这个物品
         */
        function canGetItem(item, loc, floorId) {
            if (cheat.blackItem[item.id]) {
                return false;
            }
            // 可以用于检测道具是否应该被捡起，例如如果捡起后血量超过80%则不捡起可以这么写：
            // if (item.cls === 'items') {
            //     let diff = {};
            //     const before = core.status.hero;
            //     const hero = core.clone(core.status.hero);
            //     const handler = {
            //         set(target, key, v) {
            //             diff[key] = v - (target[key] || 0);
            //             if (!diff[key]) diff[key] = void 0;
            //             return true;
            //         }
            //     };
            //     core.status.hero = new Proxy(hero, handler);

            //     eval(item.itemEffect);

            //     core.status.hero = before;
            //     window.hero = before;
            //     window.flags = before.flags;
            //     if (
            //         diff.hp &&
            //         diff.hp + core.status.hero.hp > core.status.hero.hpmax * 0.8
            //     )
            //         return false;
            // }
            return true;
        }

        /**
         * @template T
         * @param {T} v
         * @returns {v is NonNullable<T>}
         */
        function has(v) {
            if (v == null) {
                return false;
            } else if (Array.isArray(v) && v.length == 0) {
                return false;
            }
            return true;
        }

        /**
         * 广搜，搜索可以到达的需要清的怪
         * @param {string} floorId
         */
        function bfs(floorId, deep = Infinity) {
            core.extractBlocks?.(floorId);
            const objs = core.getMapBlocksObj(floorId);
            const {x, y} = core.status.hero.loc;
            /** @type {[direction, number, number][]} */
            const dir = Object.entries(core.utils.scan).map(v => [
                v[0],
                v[1].x,
                v[1].y
            ]);
            const floor = core.status.maps[floorId];

            /** @type {[number, number][]} */
            const queue = [[x, y]];
            const mapped = {
                [`${x},${y}`]: true
            };
            while (queue.length > 0 && deep > 0) {
                const [nx, ny] = queue.shift();
                dir.forEach(v => {
                    const [tx, ty] = [nx + v[1], ny + v[2]];
                    if (
                        tx < 0 ||
                        ty < 0 ||
                        tx >= floor.width ||
                        ty >= floor.height
                    ) {
                        return;
                    }
                    const loc = `${tx},${ty}`;
                    if (mapped[loc]) return;
                    const block = objs[loc];
                    mapped[loc] = true;
                    const type = judge(block, nx, ny, tx, ty, v[0], floorId);
                    if (type === false) return;
                    const {isEnemy, isItem} = type;

                    if (isEnemy) {
                        if (
                            canBattle(block.event.id, tx, ty) &&
                            !block.disable
                        ) {
                            core.battle(block.event.id, tx, ty);
                            core.updateCheckBlock();
                        } else {
                            return;
                        }
                    } else if (isItem) {
                        const item = core.material.items[block.event.id];
                        if (canGetItem(item, loc, floorId)) {
                            core.getItem(item.id ?? block.event.id, 1, tx, ty);
                        } else {
                            return;
                        }
                    }
                    // 然后判断目标点是否有地图伤害等，没有就直接添加到队列
                    const damage = core.status.checkBlock.damage[loc];
                    const ambush = core.status.checkBlock.ambush[loc];
                    const repulse = core.status.checkBlock.repulse?.[loc];

                    if (
                        (has(damage) && damage > 0) ||
                        has(ambush) ||
                        has(repulse)
                    ) {
                        return;
                    }
                    queue.push([tx, ty]);
                });
                deep--;
            }
        }

        core.auto = function () {
            const before = flags.__forbidSave__;
            // 如果勇士当前点有地图伤害，只清周围，如果有时间，直接不清了
            const {x, y} = core.status.hero.loc;
            if (!x || !y) {
                return;
            }
            const floor = core.floors[core.status.floorId];
            const loc = `${x},${y}`;
            const hasEvent = has(floor.events[loc]);
            if (hasEvent) return;
            const damage = core.status.checkBlock.damage[loc];
            const ambush = core.status.checkBlock.ambush[loc];
            const repulse = core.status.checkBlock.repulse?.[loc];
            let deep = 100;
            if ((has(damage) && damage > 0) || has(ambush) || has(repulse)) {
                deep = 1;
            }
            flags.__forbidSave__ = true;
            bfs(core.status.floorId, deep);
            flags.__forbidSave__ = before;
            core.updateStatusBar();
        };
        delete cheat.autoBattle;
    },
    afterAuto: () => {

    },
}
