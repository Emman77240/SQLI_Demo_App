from app import app
import unittest


class FlaskTestCase(unittest.TestCase):

    # Ensure that flask was properly set up
    def test_index(self):
        tester = app.test_client(self)
        response = tester.get('/login', content_type='html/text')
        self.assertEqual(response.status_code, 200)


    # Ensure that login page loads correctly 
    def test_login_page_loads(self):
        tester = app.test_client(self)
        response = tester.get('/login', content_type='html/text')
        self.assertTrue(b'Please login' in response.data)

    
    # Ensure that login behaves correctly given the right credentials
    def test_correct_login(self):
        tester = app.test_client(self)
        response = tester.post('/login', data=dict(username="admin", password="admin"), follow_redirects=True)
        self.assertIn(b'You were just logged in!', response.data)



    # Ensure that login behaves correctly given the wrong credentials
    def test_incorrect_login(self):
        tester = app.test_client(self)
        response = tester.post('/login', data=dict(username = "wrong", password = "wrong"), follow_redirects=True)
        self.assertIn(b'Invalid credentials. Please try again.', response.data)


    # Ensure that logout behaves correctly
    def test_correct_logout(self):
        tester = app.test_client(self)
        tester.post('/login', data=dict(username = "wrong", password = "wrong"), follow_redirects=True)
        response = tester.get('/logout', follow_redirects = True, content_type = 'html/text')
        self.assertTrue(b'You were just logged out!', response.data)

if __name__ == '__main__':
    unittest.main()