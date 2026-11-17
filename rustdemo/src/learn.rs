use crate::learn::List::{Cons, Nil};
use std::cell::{RefCell, RefMut};
use std::fmt::{Debug, Display, Formatter};
use std::rc::Rc;

trait LinkedList {
    fn get_next(&self) -> Option<Rc<Self>>;
    fn set_next(&self, next: Rc<Self>);
}

enum List {
    // 链表的后继需要能在 &self 下被替换，所以用 RefCell 包一层
    Cons(Rc<RefCell<i32>>, RefCell<Rc<List>>),
    Nil,
}
impl List {
    fn new(v: i32) -> List {
        Cons(Rc::new(RefCell::new(v)), RefCell::new(Rc::new(Nil)))
    }
    fn new_rc(v: i32) -> Rc<List> {
        Rc::new(Self::new(v))
    }
    fn value(&self) -> Option<RefMut<'_, i32>> {
        match self {
            Cons(v, _) => Some(v.borrow_mut()),
            Nil => None,
        }
    }
}
impl Display for List {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Cons(v, next) => f.write_fmt(format_args!("List({}) -> {}", v.borrow(), next.borrow())),
            Nil => f.write_fmt(format_args!("null")),
        }
    }
}
impl Debug for List {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self, f)
    }
}
impl LinkedList for List {
    fn get_next(&self) -> Option<Rc<Self>> {
        match self {
            Cons(_, link) => Some(Rc::clone(&link.borrow())),
            Nil => None,
        }
    }
    fn set_next(&self, next: Rc<List>) {
        if let Cons(_, link) = self {
            *link.borrow_mut() = next;
        }
    }
}
impl Drop for List {
    fn drop(&mut self) {
        if let Nil = self {
            return;
        }
        println!("List drop {}", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Weak;
    fn print_rc<T>(pre: &str, rc: &Rc<T>) {
        println!(
            "{} = strong {} weak {}",
            pre,
            Rc::strong_count(rc),
            Rc::weak_count(rc)
        )
    }
    fn print_weak<T>(pre: &str, rc: &Weak<T>) {
        println!(
            "{} = strong {} weak {}",
            pre,
            Weak::strong_count(rc),
            Weak::weak_count(rc)
        )
    }
    #[test]
    fn test_list() {
        let l1 = List::new(1);
        let l2 = List::new_rc(2);
        let l3 = List::new_rc(3);
        l1.set_next(Rc::clone(&l2));
        l2.set_next(l3);
        println!("{} l1 = {l1}", line!());
        print_rc("l2.ref", &l2);
        drop(l2);
        let l2 = l1.get_next();
        if let Some(mut v) = l1.value() {
            *v = 10
        }
        println!("{} l1.get_next = {:?}", line!(), l2);
        if let Some(l2) = l2 {
            print_rc("l2.ref", &l2);
            println!("l1.get_next = {}", l2);
            if let Some(mut v) = l2.value() {
                *v = 20;
            }
        }
        // get_next() 返回 Rc 克隆而非借用守卫，所以无需 drop 就能继续修改链表
        let l4 = Rc::new(List::new(4));
        l1.set_next(Rc::clone(&l4));
        println!("{} l1 = {}", line!(), l1);
        println!("{} l1.get_next = {:?}", line!(), l1.get_next());
        print_rc("l4.ref", &l4);
        l1.set_next(Nil.into());
        let l4w = Rc::downgrade(&l4);
        drop(l4);
        print_weak("l4.ref", &l4w);
    }
}
